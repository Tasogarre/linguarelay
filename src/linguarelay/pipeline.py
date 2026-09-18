from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import urllib.parse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import AppConfig
from .ffmpeg import (
    MediaError,
    burn_subtitles,
    decode_smoke,
    extract_clip,
    lower_frame_difference_ratio,
    media_duration,
    mux_soft_subtitles,
    normalize_audio,
    probe_summary,
    sample_cue_frames,
    split_audio,
)
from .metrics import TranscriptEvaluation, evaluate_transcript
from .models import Cue, Segment, Transcript, TranslationSegment
from .providers import (
    ASRProvider,
    DeepgramProvider,
    NativeCaptionProvider,
    OllamaTranslationProvider,
    OpenAITranslationProvider,
    OpenAIWhisperProvider,
    ProviderError,
    ProviderUnavailable,
    merge_transcripts,
    offset_transcript,
)
from .security import JobWorkspace, sha256_file
from .state import JobState
from .subtitles import (
    build_cues,
    parse_srt,
    parse_vtt,
    render_srt,
    render_vtt,
    text_coverage_matches,
    validate_cues,
)


STAGES = (
    "normalize",
    "transcribe",
    "translate",
    "subtitles",
    "render",
    "validate",
    "manifest",
)

EVALUATION_STAGES = ("prepare-eval", "evaluate-asr")
ALL_STAGES = (
    "normalize",
    "prepare-eval",
    "evaluate-asr",
    "transcribe",
    "translate",
    "subtitles",
    "render",
    "validate",
    "manifest",
)

_STAGE_DEPENDENCIES = {
    "normalize": "acquire",
    "prepare-eval": "normalize",
    "evaluate-asr": "prepare-eval",
    "transcribe": "evaluate-asr",
    "translate": "transcribe",
    "subtitles": "translate",
    "render": "subtitles",
    "validate": "render",
    "manifest": "validate",
}

_STAGE_CONTEXT_PATHS = {
    "evaluate-asr": (
        "asr/reference.json",
        "source/native.pt-BR.vtt",
        "source/native-caption-evidence.json",
    ),
    "translate": ("asr/reference.json",),
}


class StageBlocked(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = message
        super().__init__(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise StageBlocked(
            "ARTIFACT_INVALID", "A required job artifact is missing or invalid."
        ) from None
    if not isinstance(value, dict):
        raise StageBlocked(
            "ARTIFACT_INVALID", "A required job artifact has an invalid shape."
        )
    return value


def _transcript_from_reference(payload: dict[str, Any]) -> Transcript:
    value = dict(payload)
    value.setdefault("provider", "human_reference")
    value.setdefault("model", "manual_review")
    value.setdefault("source", "manual_reference")
    try:
        return Transcript.from_dict(value)
    except (KeyError, TypeError, ValueError):
        raise StageBlocked(
            "MANUAL_REFERENCE_INVALID", "The ASR reference has an invalid shape."
        ) from None


def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _number_value(token: str, language: str) -> Decimal:
    normalized = (
        token.replace(".", "").replace(",", ".")
        if language == "pt-BR"
        else token.replace(",", "")
    )
    try:
        return Decimal(normalized).normalize()
    except InvalidOperation:
        return Decimal("NaN")


def validate_translation(
    source: Sequence[Segment],
    translated: Sequence[TranslationSegment],
    *,
    glossary_terms: Sequence[str] = (),
) -> list[str]:
    violations: list[str] = []
    source_by_id = {segment.id: segment for segment in source}
    translated_by_id = {segment.id: segment for segment in translated}
    if len(source_by_id) != len(source) or len(translated_by_id) != len(translated):
        violations.append("duplicate segment identifiers")
    if set(source_by_id) != set(translated_by_id):
        violations.append("translation coverage is incomplete")
        return violations
    for segment in source:
        result = translated_by_id[segment.id]
        if not result.text.strip():
            violations.append(f"segment {segment.id}: empty translation")
        if (
            abs(result.start - segment.start) > 0.001
            or abs(result.end - segment.end) > 0.001
        ):
            violations.append(f"segment {segment.id}: timing changed")
        if not all(math.isfinite(value) for value in (result.start, result.end)):
            violations.append(f"segment {segment.id}: non-finite timing")
        if result.source_text != segment.text:
            violations.append(f"segment {segment.id}: source text changed")
        if not set(segment.uncertainty).issubset(result.uncertainty):
            violations.append(f"segment {segment.id}: source uncertainty was dropped")
        source_numbers = Counter(
            _number_value(token, "pt-BR")
            for token in re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", segment.text)
        )
        translated_numbers = Counter(
            _number_value(token, "en")
            for token in re.findall(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", result.text)
        )
        for number in sorted((source_numbers - translated_numbers).elements()):
            violations.append(f"segment {segment.id}: missing number {number}")
        for number in sorted((translated_numbers - source_numbers).elements()):
            violations.append(f"segment {segment.id}: unexpected number {number}")
        for term in glossary_terms:
            source_has_term = _contains_term(segment.text, term)
            translation_has_term = _contains_term(result.text, term)
            if source_has_term and not translation_has_term:
                violations.append(
                    f"segment {segment.id}: missing glossary term {term}"
                )
            if not source_has_term and translation_has_term:
                violations.append(
                    f"segment {segment.id}: unexpected glossary term {term}"
                )
    return violations


class Pipeline:
    def __init__(
        self,
        config: AppConfig,
        workspace: JobWorkspace,
        *,
        asr_overrides: dict[str, ASRProvider] | None = None,
        translation_override: Any | None = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.state = JobState(workspace)
        self.asr_overrides = asr_overrides or {}
        self.translation_override = translation_override

    def run(
        self,
        *,
        stages: Sequence[str] | None = None,
        resume: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        default_stages = (
            ALL_STAGES
            if self.config.section("asr").get("mode") == "evaluate"
            else STAGES
        )
        selected = list(stages or default_stages)
        unknown = [stage for stage in selected if stage not in ALL_STAGES]
        if unknown:
            raise StageBlocked(
                "UNKNOWN_STAGE", "One or more requested stages are unknown."
            )
        if dry_run:
            return selected

        executed: list[str] = []
        resume_cache: dict[str, bool] = {}
        for stage in selected:
            if resume and self._can_resume_stage(stage, cache=resume_cache):
                continue
            resume_cache.clear()
            self.state.start(stage)
            try:
                outputs = getattr(self, f"stage_{stage.replace('-', '_')}")()
            except StageBlocked as exc:
                self.state.fail(stage, exc.code)
                raise
            except ProviderUnavailable:
                self.state.fail(stage, "PROVIDER_UNAVAILABLE")
                raise StageBlocked(
                    "PROVIDER_UNAVAILABLE",
                    "A configured provider is unavailable; inspect redacted job evidence.",
                ) from None
            except ProviderError:
                self.state.fail(stage, "PROVIDER_FAILURE")
                raise StageBlocked(
                    "PROVIDER_FAILURE",
                    "A provider failed; no response body or credential was logged.",
                ) from None
            except MediaError:
                self.state.fail(stage, "MEDIA_FAILURE")
                raise StageBlocked(
                    "MEDIA_FAILURE", "A local media operation failed."
                ) from None
            except Exception:
                self.state.fail(stage, "INTERNAL_FAILURE")
                raise StageBlocked(
                    "INTERNAL_FAILURE",
                    "The stage failed at an internal boundary; no exception detail was logged.",
                ) from None
            self.state.complete(
                stage, outputs, context_sha256=self._stage_context_hash(stage)
            )
            executed.append(stage)
        return executed

    def _source_media(self) -> Path:
        candidates = [
            path
            for path in sorted(self.workspace.resolve("source").iterdir())
            if path.is_file()
            and path.name.startswith("source.")
            and path.suffix.casefold() in {".mp4", ".mkv", ".mov", ".webm", ".ts", ".m4v"}
        ]
        if len(candidates) != 1:
            raise StageBlocked(
                "SOURCE_MISSING",
                "Exactly one safely acquired source media file is required before normalization.",
            )
        return candidates[0]

    def _media_settings(self) -> dict[str, Any]:
        return self.config.section("media")

    def _stage_context_hash(self, stage: str) -> str:
        digest = hashlib.sha256()
        digest.update(stage.encode())
        if self.config.path.is_file():
            digest.update(sha256_file(self.config.path).encode())
        for relative in _STAGE_CONTEXT_PATHS.get(stage, ()):
            digest.update(relative.encode())
            path = self.workspace.resolve(relative)
            digest.update(sha256_file(path).encode() if path.is_file() else b"MISSING")
        return digest.hexdigest()

    def _can_resume_stage(
        self, stage: str, *, cache: dict[str, bool] | None = None
    ) -> bool:
        decisions = cache if cache is not None else {}
        if stage in decisions:
            return decisions[stage]
        if not self.state.can_resume(
            stage, context_sha256=self._stage_context_hash(stage)
        ):
            decisions[stage] = False
            return False
        dependency = _STAGE_DEPENDENCIES[stage]
        if stage == "transcribe" and self.config.section("asr").get("mode") == "single":
            dependency = "normalize"
        dependency_valid = (
            self.state.can_resume(dependency)
            if dependency == "acquire"
            else self._can_resume_stage(dependency, cache=decisions)
        )
        if not dependency_valid:
            decisions[stage] = False
            return False
        records = self.state.data["stages"]
        dependency_time = str(records[dependency].get("completed_at", ""))
        stage_time = str(records[stage].get("completed_at", ""))
        decisions[stage] = bool(
            dependency_time and stage_time and dependency_time <= stage_time
        )
        return decisions[stage]

    def _verified_source_receipt(self) -> tuple[Path, Path, dict[str, Any]]:
        source = self._source_media()
        discovery_path = self.workspace.resolve("source/discovery.json")
        discovery = _load_json(discovery_path)
        expected_path = source.relative_to(self.workspace.root).as_posix()
        if (
            discovery.get("status") != "PASS"
            or discovery.get("source_path") != expected_path
            or discovery.get("sha256") != sha256_file(source)
            or discovery.get("bytes") != source.stat().st_size
            or discovery.get("transport") not in {"local", "mp4", "hls", "dash"}
        ):
            raise StageBlocked(
                "SOURCE_RECEIPT_INVALID",
                "The acquired source does not match its redacted discovery receipt.",
            )
        if stat.S_IMODE(source.stat().st_mode) & 0o222:
            raise StageBlocked(
                "SOURCE_NOT_IMMUTABLE", "The acquired source must be read-only."
            )
        return source, discovery_path, discovery

    def _require_pass_artifact(self, relative: str, code: str) -> dict[str, Any]:
        payload = _load_json(self.workspace.resolve(relative))
        if payload.get("status") != "PASS":
            raise StageBlocked(code, "A required upstream quality gate is not passing.")
        return payload

    def stage_normalize(self) -> list[Path]:
        source, _, _ = self._verified_source_receipt()
        media = self._media_settings()
        destination = self.workspace.resolve("audio/normalized.wav")
        normalize_audio(
            source,
            destination,
            sample_rate=int(media.get("audio_sample_rate", 16000)),
            channels=int(media.get("audio_channels", 1)),
            ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
        )
        return [destination]

    def stage_prepare_eval(self) -> list[Path]:
        audio = self.workspace.resolve("audio/normalized.wav")
        if not audio.is_file():
            raise StageBlocked(
                "NORMALIZED_AUDIO_MISSING",
                "Normalized audio is required before ASR evaluation.",
            )
        media = self._media_settings()
        duration = media_duration(audio, ffprobe=str(media.get("ffprobe", "ffprobe")))
        positions = [
            float(item)
            for item in media.get("representative_clip_positions", [0.12, 0.5, 0.82])
        ]
        clip_seconds = min(
            float(media.get("representative_clip_seconds", 45)),
            duration / max(1, len(positions)),
        )
        records = []
        outputs: list[Path] = []
        for index, position in enumerate(positions, start=1):
            center = duration * min(1.0, max(0.0, position))
            start = min(
                max(0.0, center - clip_seconds / 2), max(0.0, duration - clip_seconds)
            )
            destination = self.workspace.resolve(f"audio/eval/clip-{index:02d}.flac")
            extract_clip(
                audio,
                destination,
                start=start,
                duration=clip_seconds,
                ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
            )
            outputs.append(destination)
            records.append(
                {
                    "ordinal": index,
                    "path": destination.relative_to(self.workspace.root).as_posix(),
                    "start": round(start, 3),
                    "duration": round(clip_seconds, 3),
                    "sha256": sha256_file(destination),
                }
            )
        metadata = self.workspace.write_json(
            "audio/eval-clips.json",
            {"schema_version": 1, "media_duration": duration, "clips": records},
        )
        return [*outputs, metadata]

    def _native_is_eligible(self, provider: dict[str, Any]) -> bool:
        evidence = self.workspace.resolve(
            str(provider.get("evidence_path", "source/native-caption-evidence.json"))
        )
        caption = self.workspace.resolve(
            str(provider.get("caption_path", "source/native.pt-BR.vtt"))
        )
        if not evidence.is_file() or not caption.is_file():
            return False
        value = _load_json(evidence)
        try:
            coverage = float(value.get("coverage_ratio", 0.0))
        except (TypeError, ValueError):
            return False
        return (
            value.get("origin") == "player_track"
            and value.get("language") == self.config.source_language
            and coverage >= 0.95
            and value.get("manually_spot_checked") is True
        )

    def _asr_provider(self, name: str) -> ASRProvider:
        if name in self.asr_overrides:
            return self.asr_overrides[name]
        asr = self.config.section("asr")
        provider = asr.get("providers", {}).get(name)
        if not isinstance(provider, dict):
            raise ProviderUnavailable("configured ASR provider definition is absent")
        kind = provider.get("kind")
        if kind == "native_caption":
            if not self._native_is_eligible(provider):
                raise ProviderUnavailable(
                    "native captions lack required trust evidence"
                )
            return NativeCaptionProvider(
                name=name,
                model="player-native",
                caption_path=self.workspace.resolve(str(provider.get("caption_path"))),
            )
        if kind == "deepgram":
            return DeepgramProvider(
                name=name,
                model=str(provider.get("model", "nova-3")),
                api_key_env=str(provider.get("api_key_env", "DEEPGRAM_API_KEY")),
            )
        if kind == "openai_whisper":
            return OpenAIWhisperProvider(
                name=name,
                model=str(provider.get("model", "whisper-1")),
                api_key_env=str(provider.get("api_key_env", "OPENAI_API_KEY")),
            )
        raise ProviderUnavailable("configured ASR provider kind is unsupported")

    def _reference(self) -> tuple[dict[str, Any], Transcript]:
        path = self.workspace.resolve("asr/reference.json")
        if not path.is_file():
            raise StageBlocked(
                "MANUAL_REFERENCE_MISSING",
                "A manually reviewed representative-clip reference is required for the ASR bake-off.",
            )
        payload = _load_json(path)
        asr = self.config.section("asr")
        if asr.get("require_manually_reviewed_reference", True):
            if (
                payload.get("manually_reviewed") is not True
                or not str(payload.get("reviewed_by", "")).strip()
            ):
                raise StageBlocked(
                    "MANUAL_REFERENCE_UNVERIFIED",
                    "The ASR reference does not contain required human-review evidence.",
                )
        transcript = _transcript_from_reference(payload)
        if not transcript.segments or not transcript.text.strip():
            raise StageBlocked(
                "MANUAL_REFERENCE_EMPTY",
                "The ASR reference contains no reviewed speech.",
            )
        return payload, transcript

    def _clip_records(self) -> list[dict[str, Any]]:
        metadata = _load_json(self.workspace.resolve("audio/eval-clips.json"))
        clips = metadata.get("clips", [])
        if not isinstance(clips, list) or not clips:
            raise StageBlocked(
                "EVAL_CLIPS_MISSING", "Representative evaluation clips are missing."
            )
        return clips

    def _transcribe_eval_clips(
        self, provider: ASRProvider, clips: list[dict[str, Any]]
    ) -> Transcript:
        if isinstance(provider, NativeCaptionProvider):
            transcript = provider.transcribe(
                self.workspace.resolve("audio/normalized.wav"),
                language=self.config.source_language,
                min_confidence=float(
                    self.config.quality.get("minimum_confidence", 0.65)
                ),
            )
            intervals = [
                (float(clip["start"]), float(clip["start"]) + float(clip["duration"]))
                for clip in clips
            ]
            transcript.segments = [
                segment
                for segment in transcript.segments
                if any(
                    segment.end > start and segment.start < end
                    for start, end in intervals
                )
            ]
            return transcript
        parts: list[Transcript] = []
        for clip in clips:
            transcript = provider.transcribe(
                self.workspace.resolve(clip["path"]),
                language=self.config.source_language,
                min_confidence=float(
                    self.config.quality.get("minimum_confidence", 0.65)
                ),
            )
            parts.append(
                offset_transcript(
                    transcript, float(clip["start"]), f"clip{clip['ordinal']}"
                )
            )
        return merge_transcripts(parts)

    def _candidate_gates(self, metrics: TranscriptEvaluation) -> list[str]:
        quality = self.config.quality
        failures = []
        if metrics.wer > float(quality.get("max_wer", 1.0)):
            failures.append("wer")
        if metrics.cer > float(quality.get("max_cer", 1.0)):
            failures.append("cer")
        if metrics.number_recall < float(quality.get("min_number_recall", 0.0)):
            failures.append("number_recall")
        if metrics.glossary_term_recall < float(
            quality.get("min_glossary_term_recall", 0.0)
        ):
            failures.append("glossary_term_recall")
        if metrics.timing_iou < float(quality.get("min_timing_iou", 0.0)):
            failures.append("timing_iou")
        return failures

    @staticmethod
    def _candidate_score(metrics: TranscriptEvaluation) -> float:
        return (
            metrics.wer
            + 0.35 * metrics.cer
            + 0.15 * (1 - metrics.glossary_term_recall)
            + 0.15 * (1 - metrics.number_recall)
            + 0.05 * (1 - metrics.punctuation_f1)
            + 0.10 * (1 - metrics.timing_iou)
        )

    def stage_evaluate_asr(self) -> list[Path]:
        reference_payload, reference = self._reference()
        clips = self._clip_records()
        asr = self.config.section("asr")
        candidates = [str(item) for item in asr.get("candidates", [])]
        primary_metric = str(asr.get("winner_primary_metric", "wer"))
        if primary_metric != "wer":
            raise StageBlocked(
                "ASR_SELECTION_POLICY_INVALID",
                "Only WER-primary ASR selection is supported.",
            )
        evidence: list[dict[str, Any]] = []
        outputs: list[Path] = []
        successful = 0
        glossary_terms = [
            str(item) for item in reference_payload.get("glossary_terms", [])
        ]
        expected_numbers = [str(item) for item in reference_payload.get("numbers", [])]
        for name in candidates:
            try:
                provider = self._asr_provider(name)
                transcript = self._transcribe_eval_clips(provider, clips)
            except ProviderUnavailable:
                evidence.append(
                    {"candidate": name, "status": "excluded", "reason": "unavailable"}
                )
                continue
            except ProviderError:
                evidence.append(
                    {
                        "candidate": name,
                        "status": "failed",
                        "reason": "provider_failure",
                    }
                )
                continue
            successful += 1
            candidate_path = self.workspace.resolve(f"asr/candidates/{name}.json")
            self.workspace.write_json(
                candidate_path.relative_to(self.workspace.root),
                transcript.to_dict(),
                validate=False,
            )
            outputs.append(candidate_path)
            metrics = evaluate_transcript(
                reference,
                transcript,
                glossary_terms=glossary_terms,
                expected_numbers=expected_numbers,
            )
            failures = self._candidate_gates(metrics)
            evidence.append(
                {
                    "candidate": name,
                    "provider_kind": transcript.source,
                    "model": transcript.model,
                    "status": "eligible" if not failures else "ineligible",
                    "metrics": metrics.to_dict(),
                    "gate_failures": failures,
                    "score": self._candidate_score(metrics),
                }
            )
        if successful < 2:
            report = self.workspace.write_json(
                "asr/evaluation.json",
                {"schema_version": 1, "status": "BLOCKED", "candidates": evidence},
            )
            outputs.append(report)
            raise StageBlocked(
                "ASR_CANDIDATES_INSUFFICIENT",
                "At least two available ASR candidates must be evaluated against the manual reference.",
            )
        passing = [item for item in evidence if item.get("status") == "eligible"]
        if not passing:
            report = self.workspace.write_json(
                "asr/evaluation.json",
                {"schema_version": 1, "status": "BLOCKED", "candidates": evidence},
            )
            outputs.append(report)
            raise StageBlocked(
                "ASR_QUALITY_GATES_FAILED",
                "No evaluated ASR candidate passed every quality gate.",
            )
        winner = min(
            passing,
            key=lambda item: (
                float(item["metrics"]["wer"]),
                0 if item["candidate"] == "native" else 1,
                float(item["score"]),
                item["candidate"],
            ),
        )
        report = self.workspace.write_json(
            "asr/evaluation.json",
            {
                "schema_version": 1,
                "status": "PASS",
                "reference": {
                    "manually_reviewed": True,
                    "segment_count": len(reference.segments),
                    "glossary_term_count": len(glossary_terms),
                    "number_count": len(expected_numbers),
                },
                "candidates": evidence,
                "selected_candidate": winner["candidate"],
                "selection_rule": "quality gates, then minimum WER; trustworthy native wins exact WER ties",
            },
        )
        outputs.append(report)
        return outputs

    def _selected_asr(self) -> str:
        asr = self.config.section("asr")
        evaluation_path = self.workspace.resolve("asr/evaluation.json")
        if asr.get("mode") == "single":
            candidates = asr.get("candidates", [])
            if isinstance(candidates, list) and len(candidates) == 1:
                selected = str(candidates[0])
                if evaluation_path.is_file():
                    evaluation = _load_json(evaluation_path)
                    if (
                        evaluation.get("status") == "PASS"
                        and evaluation.get("mode") == "single"
                        and evaluation.get("selected_candidate") == selected
                    ):
                        return selected
                self.workspace.write_json(
                    "asr/evaluation.json",
                    {
                        "schema_version": 1,
                        "status": "PASS",
                        "mode": "single",
                        "selected_candidate": selected,
                        "candidates": [
                            {"candidate": selected, "status": "selected"}
                        ],
                    },
                )
                return selected
            raise StageBlocked("ASR_SELECTION_MISSING", "Single-provider ASR is not configured.")
        evaluation = _load_json(evaluation_path)
        selected = evaluation.get("selected_candidate")
        if evaluation.get("status") != "PASS" or not isinstance(selected, str):
            raise StageBlocked(
                "ASR_SELECTION_MISSING",
                "A passing ASR bake-off is required before transcription.",
            )
        return selected

    def stage_transcribe(self) -> list[Path]:
        selected = self._selected_asr()
        provider = self._asr_provider(selected)
        audio = self.workspace.resolve("audio/normalized.wav")
        if not audio.is_file():
            raise StageBlocked(
                "NORMALIZED_AUDIO_MISSING",
                "Normalized audio is required before transcription.",
            )
        minimum_confidence = float(self.config.quality.get("minimum_confidence", 0.65))
        if isinstance(provider, NativeCaptionProvider):
            transcript = provider.transcribe(
                audio,
                language=self.config.source_language,
                min_confidence=minimum_confidence,
            )
        else:
            media = self._media_settings()
            chunk_seconds = int(media.get("asr_chunk_seconds", 600))
            chunk_dir = self.workspace.resolve("transcript/chunks")
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for stale in chunk_dir.glob("chunk-*.flac"):
                stale.unlink()
            chunks = split_audio(
                audio,
                chunk_dir / "chunk-%03d.flac",
                seconds=chunk_seconds,
                ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
            )
            parts = []
            for index, chunk in enumerate(chunks):
                part = provider.transcribe(
                    chunk,
                    language=self.config.source_language,
                    min_confidence=minimum_confidence,
                )
                parts.append(
                    offset_transcript(part, index * chunk_seconds, f"part{index + 1}")
                )
            transcript = merge_transcripts(parts)
        if not transcript.segments or not transcript.text.strip():
            raise StageBlocked(
                "TRANSCRIPT_EMPTY", "The selected ASR provider returned no speech."
            )
        media = self._media_settings()
        duration = media_duration(audio, ffprobe=str(media.get("ffprobe", "ffprobe")))
        violations: list[str] = []
        if transcript.language != self.config.source_language:
            violations.append(
                "transcript language does not match configured source language"
            )
        seen_ids: set[str] = set()
        previous_start = -1.0
        for segment in transcript.segments:
            if segment.id in seen_ids:
                violations.append(f"segment {segment.id}: duplicate identifier")
            seen_ids.add(segment.id)
            if not segment.text.strip():
                violations.append(f"segment {segment.id}: empty text")
            if not all(math.isfinite(value) for value in (segment.start, segment.end)):
                violations.append(f"segment {segment.id}: non-finite timing")
            elif (
                segment.start < 0
                or segment.end <= segment.start
                or segment.end > duration + 1.0
            ):
                violations.append(f"segment {segment.id}: invalid media bounds")
            elif segment.start < previous_start:
                violations.append(f"segment {segment.id}: out-of-order timing")
            else:
                previous_start = segment.start
            if segment.confidence is not None:
                if (
                    not math.isfinite(segment.confidence)
                    or not 0 <= segment.confidence <= 1
                ):
                    violations.append(f"segment {segment.id}: invalid confidence")
                elif (
                    segment.confidence < minimum_confidence
                    and "low_segment_confidence" not in segment.uncertainty
                ):
                    segment.uncertainty.append("low_segment_confidence")
            for word in segment.words:
                if (
                    not all(math.isfinite(value) for value in (word.start, word.end))
                    or word.start < 0
                    or word.end <= word.start
                    or word.end > duration + 1.0
                ):
                    violations.append(f"segment {segment.id}: invalid word timing")
                if not word.text.strip():
                    violations.append(f"segment {segment.id}: empty word")
                if word.confidence is not None and (
                    not math.isfinite(word.confidence) or not 0 <= word.confidence <= 1
                ):
                    violations.append(f"segment {segment.id}: invalid word confidence")
        path = self.workspace.write_json(
            "transcript/pt-BR.raw.json", transcript.to_dict(), validate=False
        )
        validation = self.workspace.write_json(
            "transcript/validation.json",
            {
                "schema_version": 1,
                "status": "BLOCKED" if violations else "PASS",
                "segment_count": len(transcript.segments),
                "uncertain_segment_count": sum(
                    bool(segment.uncertainty) for segment in transcript.segments
                ),
                "violation_count": len(violations),
                "violations": violations,
            },
        )
        if violations:
            raise StageBlocked(
                "TRANSCRIPT_VALIDATION_FAILED",
                "Transcript timing or confidence validation failed.",
            )
        return [path, validation]

    def _translation_provider(self) -> Any:
        if self.translation_override is not None:
            return self.translation_override
        translation = self.config.section("translation")
        name = str(translation.get("provider", ""))
        provider = translation.get("providers", {}).get(name)
        if not isinstance(provider, dict):
            raise ProviderUnavailable(
                "configured translation provider definition is absent"
            )
        kind = provider.get("kind")
        if kind == "openai_responses":
            return OpenAITranslationProvider(
                name=name,
                model=str(provider.get("model", "gpt-5-mini")),
                api_key_env=str(provider.get("api_key_env", "OPENAI_API_KEY")),
                reasoning_effort=str(provider.get("reasoning_effort", "low")),
            )
        if kind == "ollama":
            endpoint = str(provider.get("endpoint", "http://127.0.0.1:11434/api/chat"))
            parsed = urllib.parse.urlsplit(endpoint)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ProviderUnavailable(
                    "local translation endpoint must use loopback"
                )
            return OllamaTranslationProvider(
                name=name,
                model=str(provider.get("model", "")),
                endpoint=endpoint,
            )
        raise ProviderUnavailable("configured translation provider kind is unsupported")

    def _glossary_terms(self) -> list[str]:
        reference_path = self.workspace.resolve("asr/reference.json")
        if not reference_path.is_file():
            return []
        payload = _load_json(reference_path)
        return [str(item) for item in payload.get("glossary_terms", [])]

    def stage_translate(self) -> list[Path]:
        self._require_pass_artifact(
            "transcript/validation.json", "TRANSCRIPT_NOT_VALIDATED"
        )
        transcript_path = self.workspace.resolve("transcript/pt-BR.raw.json")
        if not transcript_path.is_file():
            raise StageBlocked(
                "TRANSCRIPT_MISSING",
                "The pt-BR transcript is required before translation.",
            )
        try:
            transcript = Transcript.from_dict(_load_json(transcript_path))
        except (KeyError, TypeError, ValueError):
            raise StageBlocked(
                "TRANSCRIPT_INVALID", "The pt-BR transcript has an invalid shape."
            ) from None
        provider = self._translation_provider()
        translation = self.config.section("translation")
        batch_size = max(1, int(translation.get("batch_size", 40)))
        glossary_terms = self._glossary_terms()
        translated: list[TranslationSegment] = []
        for index in range(0, len(transcript.segments), batch_size):
            batch = transcript.segments[index : index + batch_size]
            translated.extend(
                provider.translate(
                    batch,
                    source_language=self.config.source_language,
                    target_language=self.config.target_language,
                    glossary_terms=glossary_terms,
                )
            )
        violations = validate_translation(
            transcript.segments, translated, glossary_terms=glossary_terms
        )
        coverage = (
            len(translated) / len(transcript.segments) if transcript.segments else 0.0
        )
        if coverage < float(
            self.config.quality.get("minimum_translation_coverage", 1.0)
        ):
            violations.append("translation coverage is below configured minimum")
        if violations:
            self.workspace.write_json(
                "translation/validation.json",
                {
                    "status": "BLOCKED",
                    "violation_count": len(violations),
                    "violations": violations,
                },
            )
            raise StageBlocked(
                "TRANSLATION_VALIDATION_FAILED", "Translation integrity checks failed."
            )
        provider_name = str(getattr(provider, "name", "configured"))
        model_name = str(getattr(provider, "model", "configured"))
        output = self.workspace.write_json(
            "translation/en.json",
            {
                "schema_version": 1,
                "source_language": self.config.source_language,
                "target_language": self.config.target_language,
                "provider": provider_name,
                "model": model_name,
                "segments": [segment.to_dict() for segment in translated],
            },
            validate=False,
        )
        validation = self.workspace.write_json(
            "translation/validation.json",
            {"status": "PASS", "coverage": coverage, "violation_count": 0},
        )
        return [output, validation]

    def _translations(self) -> list[TranslationSegment]:
        path = self.workspace.resolve("translation/en.json")
        if not path.is_file():
            raise StageBlocked(
                "TRANSLATION_MISSING",
                "The English translation is required before subtitles.",
            )
        payload = _load_json(path)
        try:
            return [
                TranslationSegment.from_dict(item)
                for item in payload.get("segments", [])
            ]
        except (KeyError, TypeError, ValueError):
            raise StageBlocked(
                "TRANSLATION_INVALID", "The English translation has an invalid shape."
            ) from None

    def stage_subtitles(self) -> list[Path]:
        self._require_pass_artifact(
            "translation/validation.json", "TRANSLATION_NOT_VALIDATED"
        )
        source = self._source_media()
        media = self._media_settings()
        duration = media_duration(source, ffprobe=str(media.get("ffprobe", "ffprobe")))
        translations = self._translations()
        cues = build_cues(
            translations, self.config.subtitle_policy, media_duration=duration
        )
        violations = validate_cues(
            cues, self.config.subtitle_policy, media_duration=duration
        )
        if not text_coverage_matches(translations, cues):
            violations.append("subtitle text coverage is incomplete")
        if violations:
            self.workspace.write_json(
                "subtitles/validation.json",
                {
                    "status": "BLOCKED",
                    "violation_count": len(violations),
                    "violations": violations,
                },
            )
            raise StageBlocked(
                "SUBTITLE_POLICY_FAILED",
                "Generated subtitles failed readability or timing gates.",
            )
        srt = self.workspace.resolve("subtitles/en.srt")
        vtt = self.workspace.resolve("subtitles/en.vtt")
        srt.write_text(render_srt(cues), encoding="utf-8")
        vtt.write_text(render_vtt(cues), encoding="utf-8")
        os.chmod(srt, 0o600)
        os.chmod(vtt, 0o600)
        cue_data = self.workspace.write_json(
            "subtitles/cues.json",
            {"schema_version": 1, "cues": [cue.to_dict() for cue in cues]},
            validate=False,
        )
        validation = self.workspace.write_json(
            "subtitles/validation.json",
            {"status": "PASS", "cue_count": len(cues), "violation_count": 0},
        )
        return [srt, vtt, cue_data, validation]

    def _cues(self) -> list[Cue]:
        path = self.workspace.resolve("subtitles/en.srt")
        if not path.is_file():
            raise StageBlocked(
                "SUBTITLES_MISSING",
                "Validated subtitles are required before rendering.",
            )
        try:
            cues = parse_srt(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise StageBlocked(
                "SUBTITLES_INVALID", "The English subtitle file is invalid."
            ) from None
        if not cues:
            raise StageBlocked("SUBTITLES_EMPTY", "The English subtitle file is empty.")
        return cues

    def stage_render(self) -> list[Path]:
        self._require_pass_artifact(
            "subtitles/validation.json", "SUBTITLES_NOT_VALIDATED"
        )
        source = self._source_media()
        srt = self.workspace.resolve("subtitles/en.srt")
        cues = self._cues()
        media = self._media_settings()
        render = self.config.section("render")
        preferred = str(render.get("soft_container", "mp4")).casefold()
        soft_paths = {
            container: self.workspace.resolve(f"output/soft-subtitled.{container}")
            for container in ("mp4", "mkv")
        }
        for path in soft_paths.values():
            path.unlink(missing_ok=True)
        soft: Path | None = None
        for container in (preferred, "mkv" if preferred == "mp4" else "mp4"):
            candidate = soft_paths[container]
            try:
                mux_soft_subtitles(
                    source, srt, candidate, ffmpeg=str(media.get("ffmpeg", "ffmpeg"))
                )
            except MediaError:
                candidate.unlink(missing_ok=True)
                continue
            soft = candidate
            break
        if soft is None:
            raise MediaError(
                "no configured soft-subtitle container accepted the source streams"
            )
        burned = self.workspace.resolve("output/burned-in.mp4")
        burn_subtitles(
            source,
            cues,
            burned,
            self.config.subtitle_policy,
            work_dir=self.workspace.resolve("output/.overlay-work"),
            ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
            ffprobe=str(media.get("ffprobe", "ffprobe")),
            video_codec=str(render.get("video_codec", "libx264")),
            audio_codec=str(render.get("audio_codec", "copy")),
            crf=int(render.get("crf", 18)),
            preset=str(render.get("preset", "medium")),
        )
        return [soft, burned]

    def _soft_output(self) -> Path:
        outputs = [
            path
            for path in (
                self.workspace.resolve("output/soft-subtitled.mp4"),
                self.workspace.resolve("output/soft-subtitled.mkv"),
            )
            if path.is_file()
        ]
        if len(outputs) != 1:
            raise StageBlocked(
                "SOFT_OUTPUT_MISSING", "Exactly one soft-subtitle output is required."
            )
        return outputs[0]

    def stage_validate(self) -> list[Path]:
        self._require_pass_artifact(
            "transcript/validation.json", "TRANSCRIPT_NOT_VALIDATED"
        )
        self._require_pass_artifact(
            "translation/validation.json", "TRANSLATION_NOT_VALIDATED"
        )
        self._require_pass_artifact(
            "subtitles/validation.json", "SUBTITLES_NOT_VALIDATED"
        )
        source, _, _ = self._verified_source_receipt()
        soft = self._soft_output()
        burned = self.workspace.resolve("output/burned-in.mp4")
        if not burned.is_file():
            raise StageBlocked(
                "BURNED_OUTPUT_MISSING",
                "The burned-in output is required for validation.",
            )
        media = self._media_settings()
        ffprobe = str(media.get("ffprobe", "ffprobe"))
        source_summary = probe_summary(source, ffprobe=ffprobe)
        soft_summary = probe_summary(soft, ffprobe=ffprobe)
        burned_summary = probe_summary(burned, ffprobe=ffprobe)
        cues = self._cues()
        violations = validate_cues(
            cues,
            self.config.subtitle_policy,
            media_duration=float(source_summary["duration"]),
        )
        source_types = {
            stream.get("codec_type") for stream in source_summary["streams"]
        }
        soft_types = {stream.get("codec_type") for stream in soft_summary["streams"]}
        burned_types = {
            stream.get("codec_type") for stream in burned_summary["streams"]
        }
        if "video" not in soft_types:
            violations.append("soft output has no video stream")
        if "subtitle" not in soft_types:
            violations.append("soft output has no subtitle stream")
        if not any(
            stream.get("codec_type") == "subtitle" and stream.get("language") == "eng"
            for stream in soft_summary["streams"]
        ):
            violations.append("soft output has no English-tagged subtitle stream")
        if "video" not in burned_types:
            violations.append("burned output has no video stream")
        if "audio" in source_types and (
            "audio" not in soft_types or "audio" not in burned_types
        ):
            violations.append("one or more outputs dropped source audio")
        for codec_type in ("video", "audio"):
            source_codec = next(
                (
                    stream.get("codec_name")
                    for stream in source_summary["streams"]
                    if stream.get("codec_type") == codec_type
                ),
                None,
            )
            soft_codec = next(
                (
                    stream.get("codec_name")
                    for stream in soft_summary["streams"]
                    if stream.get("codec_type") == codec_type
                ),
                None,
            )
            if source_codec and source_codec != soft_codec:
                violations.append(
                    f"soft output did not preserve the source {codec_type} codec"
                )
        vtt_path = self.workspace.resolve("subtitles/en.vtt")
        try:
            vtt_cues = (
                parse_vtt(vtt_path.read_text(encoding="utf-8"))
                if vtt_path.is_file()
                else []
            )
        except (OSError, ValueError):
            vtt_cues = []
        if vtt_cues != cues:
            violations.append("VTT syntax or cue parity failed")
        max_drift = float(self.config.quality.get("max_duration_drift_seconds", 0.35))
        drift = {
            "soft": abs(
                float(soft_summary["duration"]) - float(source_summary["duration"])
            ),
            "burned": abs(
                float(burned_summary["duration"]) - float(source_summary["duration"])
            ),
        }
        if drift["soft"] > max_drift or drift["burned"] > max_drift:
            violations.append("output duration drift exceeds configured maximum")
        smoke_seconds = float(self.config.quality.get("smoke_decode_seconds", 8))
        decode = {
            "soft": decode_smoke(
                soft, seconds=smoke_seconds, ffmpeg=str(media.get("ffmpeg", "ffmpeg"))
            ),
            "burned": decode_smoke(
                burned, seconds=smoke_seconds, ffmpeg=str(media.get("ffmpeg", "ffmpeg"))
            ),
        }
        if not all(decode.values()):
            violations.append("one or more output decode smoke tests failed")
        frames = sample_cue_frames(
            burned,
            cues,
            self.workspace.resolve("validation/frames"),
            count=int(self.config.quality.get("frame_sample_count", 3)),
            ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
        )
        source_frames = sample_cue_frames(
            source,
            cues,
            self.workspace.resolve("validation/source-frames"),
            count=int(self.config.quality.get("frame_sample_count", 3)),
            ffmpeg=str(media.get("ffmpeg", "ffmpeg")),
        )
        frame_sampling_valid = (
            bool(frames)
            and len(frames) == len(source_frames)
            and all(
                frame.is_file() and frame.stat().st_size > 0
                for frame in [*frames, *source_frames]
            )
        )
        if not frame_sampling_valid:
            violations.append("cue frame sampling failed")
        try:
            difference_ratios = (
                [
                    lower_frame_difference_ratio(reference, candidate)
                    for reference, candidate in zip(source_frames, frames, strict=True)
                ]
                if frame_sampling_valid
                else []
            )
        except MediaError:
            difference_ratios = []
            violations.append("sampled frame comparison failed")
        minimum_difference = float(
            self.config.quality.get("minimum_burned_frame_difference_ratio", 0.002)
        )
        if not difference_ratios or any(
            ratio < minimum_difference for ratio in difference_ratios
        ):
            violations.append("burned cue frames lack visible overlay evidence")
        report = self.workspace.write_json(
            "validation/report.json",
            {
                "schema_version": 1,
                "quality_status": "PASS" if not violations else "BLOCKED",
                "source": source_summary,
                "soft_output": soft_summary,
                "burned_output": burned_summary,
                "duration_drift_seconds": drift,
                "decode_smoke": decode,
                "sampled_frames": [
                    {
                        "path": frame.relative_to(self.workspace.root).as_posix(),
                        "sha256": sha256_file(frame),
                    }
                    for frame in frames
                ],
                "source_sampled_frames": [
                    {
                        "path": frame.relative_to(self.workspace.root).as_posix(),
                        "sha256": sha256_file(frame),
                    }
                    for frame in source_frames
                ],
                "burned_lower_frame_difference_ratio": difference_ratios,
                "violations": violations,
            },
        )
        if violations:
            raise StageBlocked(
                "OUTPUT_VALIDATION_FAILED",
                "Final output validation found one or more failures.",
            )
        return [report, *frames, *source_frames]

    @staticmethod
    def _tool_version(command: str) -> str:
        result = subprocess.run(
            [command, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        first = (
            result.stdout.splitlines()[0]
            if result.returncode == 0 and result.stdout
            else "unavailable"
        )
        return first[:160]

    def stage_manifest(self) -> list[Path]:
        validation = _load_json(self.workspace.resolve("validation/report.json"))
        if validation.get("quality_status") != "PASS":
            raise StageBlocked(
                "QUALITY_NOT_PASSING",
                "A passing validation report is required for the manifest.",
            )
        evaluation = _load_json(self.workspace.resolve("asr/evaluation.json"))
        transcript = _load_json(self.workspace.resolve("transcript/pt-BR.raw.json"))
        translation = _load_json(self.workspace.resolve("translation/en.json"))
        source, discovery_path, discovery = self._verified_source_receipt()
        status_artifacts = {
            "asr/evaluation.json": evaluation,
            "transcript/validation.json": _load_json(
                self.workspace.resolve("transcript/validation.json")
            ),
            "translation/validation.json": _load_json(
                self.workspace.resolve("translation/validation.json")
            ),
            "subtitles/validation.json": _load_json(
                self.workspace.resolve("subtitles/validation.json")
            ),
        }
        if any(
            payload.get("status") != "PASS" for payload in status_artifacts.values()
        ):
            raise StageBlocked(
                "QUALITY_NOT_PASSING", "An intermediate quality gate is not passing."
            )
        required = [
            source,
            discovery_path,
            *[self.workspace.resolve(path) for path in status_artifacts],
            self.workspace.resolve("transcript/pt-BR.raw.json"),
            self.workspace.resolve("translation/en.json"),
            self.workspace.resolve("subtitles/en.srt"),
            self.workspace.resolve("subtitles/en.vtt"),
            self._soft_output(),
            self.workspace.resolve("output/burned-in.mp4"),
            self.workspace.resolve("validation/report.json"),
        ]
        frame_evidence: dict[str, list[dict[str, Any]]] = {}
        for field in ("sampled_frames", "source_sampled_frames"):
            frames = validation.get(field)
            if not isinstance(frames, list) or not frames:
                raise StageBlocked(
                    "VALIDATION_EVIDENCE_INVALID",
                    "Frame validation evidence is invalid.",
                )
            frame_evidence[field] = frames
            for frame in frames:
                if not isinstance(frame, dict):
                    raise StageBlocked(
                        "VALIDATION_EVIDENCE_INVALID",
                        "Frame validation evidence is invalid.",
                    )
                frame_path = self.workspace.resolve(str(frame.get("path", "")))
                if not frame_path.is_file() or frame.get("sha256") != sha256_file(
                    frame_path
                ):
                    raise StageBlocked(
                        "VALIDATION_EVIDENCE_INVALID",
                        "Frame validation evidence is invalid.",
                    )
                required.append(frame_path)
        ratios = validation.get("burned_lower_frame_difference_ratio")
        minimum_difference = float(
            self.config.quality.get("minimum_burned_frame_difference_ratio", 0.002)
        )
        if (
            not isinstance(ratios, list)
            or len(ratios) != len(frame_evidence["sampled_frames"])
            or len(ratios) != len(frame_evidence["source_sampled_frames"])
        ):
            raise StageBlocked(
                "VALIDATION_EVIDENCE_INVALID", "Frame validation evidence is invalid."
            )
        try:
            ratio_values = [float(ratio) for ratio in ratios]
        except (TypeError, ValueError):
            raise StageBlocked(
                "VALIDATION_EVIDENCE_INVALID", "Frame validation evidence is invalid."
            ) from None
        if any(
            not math.isfinite(ratio) or ratio < minimum_difference
            for ratio in ratio_values
        ):
            raise StageBlocked(
                "VALIDATION_EVIDENCE_INVALID", "Frame validation evidence is invalid."
            )
        if any(not path.is_file() for path in required):
            raise StageBlocked(
                "COMPLETION_ARTIFACT_MISSING",
                "A completion-contract artifact is missing.",
            )
        media = self._media_settings()
        artifacts = [
            {
                "role": path.stem,
                "path": path.relative_to(self.workspace.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": discovery["sha256"] if path == source else sha256_file(path),
            }
            for path in required
        ]
        manifest = self.workspace.write_json(
            "manifest.json",
            {
                "schema_version": 1,
                "pipeline_version": __version__,
                "quality_status": "PASS",
                "job_id": self.workspace.root.name,
                "languages": {
                    "source": self.config.source_language,
                    "target": self.config.target_language,
                },
                "asr": {
                    "selected_candidate": evaluation.get("selected_candidate"),
                    "provider": transcript.get("provider"),
                    "model": transcript.get("model"),
                    "candidate_count": len(
                        [
                            item
                            for item in evaluation.get("candidates", [])
                            if item.get("metrics")
                        ]
                    ),
                },
                "translation": {
                    "provider": translation.get("provider"),
                    "model": translation.get("model"),
                },
                "subtitle_policy": asdict(self.config.subtitle_policy),
                "tools": {
                    "ffmpeg": self._tool_version(str(media.get("ffmpeg", "ffmpeg"))),
                    "ffprobe": self._tool_version(str(media.get("ffprobe", "ffprobe"))),
                },
                "artifacts": artifacts,
            },
        )
        return [manifest]

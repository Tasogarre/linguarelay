from __future__ import annotations

import copy
import math
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when configuration violates the pipeline contract."""


@dataclass(frozen=True, slots=True)
class SubtitlePolicy:
    max_chars_per_line: int
    max_lines: int
    max_chars_per_second: float
    min_cue_seconds: float
    max_cue_seconds: float
    minimum_gap_seconds: float
    font_path: str
    font_size_ratio: float
    bottom_margin_ratio: float


@dataclass(slots=True)
class AppConfig:
    path: Path
    repo_root: Path
    data: dict[str, Any]
    jobs_root: Path
    source_language: str
    target_language: str
    subtitle_policy: SubtitlePolicy

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"configuration section {name!r} must be a table")
        return copy.deepcopy(value)

    @property
    def quality(self) -> dict[str, Any]:
        return self.section("quality")


def _resolve_config_path(raw: str, config_path: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not expanded.is_absolute():
        expanded = config_path.parent / expanded
    return expanded.resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _required_table(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"missing [{name}] table")
    return value


def load_config(path: Path | str, *, repo_root: Path | None = None) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(
            f"unable to load configuration: {type(exc).__name__}"
        ) from None

    if data.get("schema_version") != 1:
        raise ConfigError("unsupported or missing schema_version")

    repository = (repo_root or Path.cwd()).resolve()
    paths = _required_table(data, "paths")
    languages = _required_table(data, "languages")
    subtitles = _required_table(data, "subtitles")

    try:
        jobs_root = _resolve_config_path(str(paths["jobs_root"]), config_path)
        source_language = str(languages["source"])
        target_language = str(languages["target"])
    except KeyError as exc:
        raise ConfigError(
            f"missing required configuration key: {exc.args[0]}"
        ) from None

    if _is_relative_to(jobs_root, repository):
        raise ConfigError("jobs_root must be outside the repository")
    if (source_language, target_language) != ("pt-BR", "en"):
        raise ConfigError("version 0.1 supports only pt-BR source and en target")
    media = _required_table(data, "media")
    asr = _required_table(data, "asr")
    translation = _required_table(data, "translation")
    quality = _required_table(data, "quality")
    render = _required_table(data, "render")

    positions = media.get("representative_clip_positions")
    try:
        normalized_positions = (
            [float(item) for item in positions] if isinstance(positions, list) else []
        )
    except (TypeError, ValueError):
        normalized_positions = []
    if (
        not normalized_positions
        or len(set(normalized_positions)) != len(normalized_positions)
        or any(not 0 <= item <= 1 for item in normalized_positions)
    ):
        raise ConfigError(
            "representative clip positions must be unique values from 0 to 1"
        )
    candidates = asr.get("candidates")
    candidate_names = (
        [str(item) for item in candidates] if isinstance(candidates, list) else []
    )
    asr_mode = str(asr.get("mode", "evaluate"))
    invalid_count = (
        len(candidate_names) != 1
        if asr_mode == "single"
        else len(candidate_names) < 2
    )
    if (
        invalid_count
        or any(not name.strip() for name in candidate_names)
        or len(set(candidate_names)) != len(candidate_names)
    ):
        raise ConfigError(
            "single ASR mode requires one provider; evaluate mode requires at least two"
        )
    if asr_mode not in {"single", "evaluate"}:
        raise ConfigError("ASR mode must be single or evaluate")
    asr_providers = asr.get("providers")
    if not isinstance(asr_providers, dict) or any(
        not isinstance(asr_providers.get(name), dict) for name in candidate_names
    ):
        raise ConfigError("configured ASR provider is missing")
    if asr_mode == "evaluate" and asr.get("winner_primary_metric", "wer") != "wer":
        raise ConfigError("ASR winner_primary_metric must be wer")
    selected_translation = str(translation.get("provider", ""))
    translation_providers = translation.get("providers")
    if not isinstance(translation_providers, dict) or not isinstance(
        translation_providers.get(selected_translation), dict
    ):
        raise ConfigError("configured translation provider is missing")
    try:
        if (
            int(media.get("audio_sample_rate", 0)) <= 0
            or int(media.get("audio_channels", 0)) <= 0
        ):
            raise ValueError
        representative_seconds = float(media.get("representative_clip_seconds", 0))
        if (
            int(media.get("asr_chunk_seconds", 0)) <= 0
            or not math.isfinite(representative_seconds)
            or representative_seconds <= 0
        ):
            raise ValueError
        if int(translation.get("batch_size", 0)) <= 0:
            raise ValueError
        bounded_quality = (
            "min_number_recall",
            "min_glossary_term_recall",
            "min_timing_iou",
            "minimum_translation_coverage",
            "minimum_confidence",
            "minimum_burned_frame_difference_ratio",
        )
        bounded_values = [float(quality[name]) for name in bounded_quality]
        other_quality = [
            float(quality["max_wer"]),
            float(quality["max_cer"]),
            float(quality["max_duration_drift_seconds"]),
            float(quality["smoke_decode_seconds"]),
        ]
        if any(not math.isfinite(value) for value in [*bounded_values, *other_quality]):
            raise ValueError
        if any(not 0 <= value <= 1 for value in bounded_values):
            raise ValueError
        if other_quality[0] < 0 or other_quality[1] < 0 or other_quality[2] < 0:
            raise ValueError
        if int(quality["frame_sample_count"]) <= 0 or other_quality[3] <= 0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ConfigError("media, translation, or quality values are invalid") from None

    try:
        policy = SubtitlePolicy(
            max_chars_per_line=int(subtitles["max_chars_per_line"]),
            max_lines=int(subtitles["max_lines"]),
            max_chars_per_second=float(subtitles["max_chars_per_second"]),
            min_cue_seconds=float(subtitles["min_cue_seconds"]),
            max_cue_seconds=float(subtitles["max_cue_seconds"]),
            minimum_gap_seconds=float(subtitles["minimum_gap_seconds"]),
            font_path=str(
                subtitles.get(
                    "font_path", ""
                )
            ),
            font_size_ratio=float(subtitles.get("font_size_ratio", 0.046)),
            bottom_margin_ratio=float(subtitles.get("bottom_margin_ratio", 0.075)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid subtitle policy: {type(exc).__name__}") from None

    if not (1 <= policy.max_chars_per_line <= 80 and 1 <= policy.max_lines <= 3):
        raise ConfigError("subtitle line limits are outside supported bounds")
    if not (0 < policy.min_cue_seconds <= policy.max_cue_seconds):
        raise ConfigError("subtitle cue duration policy is invalid")
    if policy.max_chars_per_second <= 0 or policy.minimum_gap_seconds < 0:
        raise ConfigError("subtitle timing policy is invalid")
    formats = {str(item).casefold() for item in subtitles.get("formats", [])}
    if formats != {"srt", "vtt"}:
        raise ConfigError("subtitles must include both SRT and VTT")
    if str(render.get("soft_container", "mp4")).casefold() not in {"mp4", "mkv"}:
        raise ConfigError("soft subtitle container must be mp4 or mkv")

    return AppConfig(
        path=config_path,
        repo_root=repository,
        data=data,
        jobs_root=jobs_root,
        source_language=source_language,
        target_language=target_language,
        subtitle_policy=policy,
    )

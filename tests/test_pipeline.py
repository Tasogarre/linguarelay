import json
import os
from pathlib import Path

import pytest

from linguarelay.config import load_config
from linguarelay.metrics import TranscriptEvaluation
from linguarelay.models import Segment, Transcript, TranslationSegment
from linguarelay.pipeline import ALL_STAGES, Pipeline, StageBlocked, validate_translation
from linguarelay.security import JobWorkspace, sha256_file
from linguarelay.state import JobState


def _config(tmp_path: Path):
    path = tmp_path / "pipeline.toml"
    baseline = Path("config/example.toml").read_text(encoding="utf-8")
    path.write_text(
        baseline.replace(
            'jobs_root = "~/.local/share/linguarelay/jobs"',
            f'jobs_root = "{tmp_path / "external-jobs"}"',
        ),
        encoding="utf-8",
    )
    return load_config(path, repo_root=tmp_path / "repo")


def test_dry_run_is_non_mutating_and_lists_explicit_stages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = JobWorkspace.create(
        config.jobs_root, "dry-run", repo_root=config.repo_root
    )
    pipeline = Pipeline(config, workspace)

    plan = pipeline.run(stages=["normalize", "prepare-eval"], dry_run=True)

    assert plan == ["normalize", "prepare-eval"]
    assert not (workspace.root / "audio/normalized.wav").exists()


def test_evaluate_mode_default_run_includes_evaluation_stages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data["asr"]["mode"] = "evaluate"
    config.data["asr"]["candidates"] = ["first", "second"]
    workspace = JobWorkspace.create(
        config.jobs_root, "evaluate-dry-run", repo_root=config.repo_root
    )

    assert Pipeline(config, workspace).run(dry_run=True) == list(ALL_STAGES)


def test_missing_source_blocks_with_stable_code(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = JobWorkspace.create(
        config.jobs_root, "missing-source", repo_root=config.repo_root
    )
    pipeline = Pipeline(config, workspace)

    with pytest.raises(StageBlocked) as captured:
        pipeline.run(stages=["normalize"])

    assert captured.value.code == "SOURCE_MISSING"
    state = json.loads((workspace.root / "state.json").read_text())
    assert state["stages"]["normalize"]["code"] == "SOURCE_MISSING"


def test_translation_validator_rejects_missing_numbers_and_terms() -> None:
    source = [Segment(id="s1", start=0.0, end=2.0, text="Ana processou 42 exemplos.")]
    translated = [
        TranslationSegment(
            id="s1",
            start=0.0,
            end=2.0,
            source_text=source[0].text,
            text="She processed examples.",
        )
    ]

    violations = validate_translation(source, translated, glossary_terms=["Ana"])

    assert "segment s1: missing number 42" in violations
    assert "segment s1: missing glossary term Ana" in violations


def test_translation_validator_rejects_added_numbers_and_dropped_uncertainty() -> None:
    source = [
        Segment(
            id="s1",
            start=0.0,
            end=2.0,
            text="Exemplo incerto.",
            uncertainty=["low_segment_confidence"],
        )
    ]
    translated = [
        TranslationSegment(
            id="s1",
            start=0.0,
            end=2.0,
            source_text=source[0].text,
            text="Uncertain example 99.",
        )
    ]

    violations = validate_translation(source, translated)

    assert "segment s1: unexpected number 99" in violations
    assert "segment s1: source uncertainty was dropped" in violations


def test_translation_validator_accepts_locale_number_formatting() -> None:
    source = [
        Segment(id="s1", start=0.0, end=2.0, text="O total é 1.500,50.")
    ]
    translated = [
        TranslationSegment(
            id="s1",
            start=0.0,
            end=2.0,
            source_text=source[0].text,
            text="The total is 1,500.50.",
        )
    ]

    assert validate_translation(source, translated) == []


def test_resume_reports_only_stages_that_execute(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = JobWorkspace.create(
        config.jobs_root, "resume", repo_root=config.repo_root
    )
    source = workspace.resolve("source/source.mp4")
    source.write_bytes(b"synthetic-source")
    os.chmod(source, 0o400)
    discovery = workspace.write_json(
        "source/discovery.json",
        {
            "status": "PASS",
            "transport": "mp4",
            "source_path": "source/source.mp4",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
    )
    JobState(workspace).complete("acquire", [source, discovery])
    output = workspace.resolve("audio/already-complete.txt")
    output.write_text("synthetic", encoding="utf-8")
    pipeline = Pipeline(config, workspace)
    pipeline.state.complete(
        "normalize",
        [output],
        context_sha256=pipeline._stage_context_hash("normalize"),
    )

    assert pipeline.run(stages=["normalize"], resume=True) == []


def test_resume_invalidates_downstream_when_acquisition_is_newer(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    workspace = JobWorkspace.create(
        config.jobs_root, "resume-order", repo_root=config.repo_root
    )
    source = workspace.resolve("source/source.mp4")
    source.write_bytes(b"synthetic-source")
    discovery = workspace.write_json("source/discovery.json", {"status": "PASS"})
    state = JobState(workspace)
    state.complete("acquire", [source, discovery])
    normalized = workspace.resolve("audio/normalized.wav")
    normalized.write_bytes(b"synthetic-audio")
    pipeline = Pipeline(config, workspace)
    pipeline.state.complete(
        "normalize",
        [normalized],
        context_sha256=pipeline._stage_context_hash("normalize"),
    )
    pipeline.state.complete("acquire", [source, discovery])

    assert not pipeline._can_resume_stage("normalize")


class _EvaluationProvider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = "fixture"

    def transcribe(self, audio_path, *, language, min_confidence):
        del audio_path, min_confidence
        return Transcript(
            language=language,
            provider=self.name,
            model=self.model,
            segments=[Segment(id="s1", start=0.0, end=1.0, text=self.name)],
        )


def test_asr_winner_uses_wer_before_weighted_tiebreakers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    config.data["asr"]["candidates"] = ["lower_wer", "lower_weighted_score"]
    workspace = JobWorkspace.create(
        config.jobs_root, "wer-primary", repo_root=config.repo_root
    )
    workspace.write_json(
        "asr/reference.json",
        {
            "language": "pt-BR",
            "manually_reviewed": True,
            "reviewed_by": "synthetic-test",
            "segments": [{"id": "r1", "start": 0.0, "end": 1.0, "text": "referência"}],
        },
        validate=False,
    )
    workspace.write_json(
        "audio/eval-clips.json",
        {
            "clips": [
                {
                    "ordinal": 1,
                    "path": "audio/eval/clip-01.flac",
                    "start": 0.0,
                    "duration": 1.0,
                }
            ]
        },
    )

    def fake_evaluation(reference, candidate, **kwargs):
        del reference, kwargs
        if candidate.provider == "lower_wer":
            return TranscriptEvaluation(0.05, 0.14, 1.0, 1.0, 0.0, 0.5)
        return TranscriptEvaluation(0.06, 0.0, 1.0, 1.0, 1.0, 1.0)

    monkeypatch.setattr("linguarelay.pipeline.evaluate_transcript", fake_evaluation)
    pipeline = Pipeline(
        config,
        workspace,
        asr_overrides={
            "lower_wer": _EvaluationProvider("lower_wer"),
            "lower_weighted_score": _EvaluationProvider("lower_weighted_score"),
        },
    )

    pipeline.stage_evaluate_asr()
    report = json.loads(
        workspace.resolve("asr/evaluation.json").read_text(encoding="utf-8")
    )

    assert report["selected_candidate"] == "lower_wer"


def test_single_asr_selection_refreshes_stale_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.data["asr"]["candidates"] = ["fixture"]
    config.data["asr"]["providers"]["fixture"] = {"kind": "openai_whisper"}
    workspace = JobWorkspace.create(
        config.jobs_root, "single-selection", repo_root=config.repo_root
    )
    workspace.write_json(
        "asr/evaluation.json",
        {
            "status": "PASS",
            "mode": "single",
            "selected_candidate": "stale",
            "candidates": [{"candidate": "stale", "status": "selected"}],
        },
    )

    selected = Pipeline(config, workspace)._selected_asr()

    assert selected == "fixture"
    evaluation = json.loads(
        workspace.resolve("asr/evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["selected_candidate"] == "fixture"


def test_manifest_rejects_empty_frame_validation_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = JobWorkspace.create(
        config.jobs_root, "manifest-evidence", repo_root=config.repo_root
    )
    source = workspace.resolve("source/source.mp4")
    source.write_bytes(b"synthetic-source")
    os.chmod(source, 0o400)
    workspace.write_json(
        "source/discovery.json",
        {
            "status": "PASS",
            "transport": "mp4",
            "source_path": "source/source.mp4",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
    )
    workspace.write_json("asr/evaluation.json", {"status": "PASS", "candidates": []})
    workspace.write_json("transcript/pt-BR.raw.json", {}, validate=False)
    workspace.write_json("transcript/validation.json", {"status": "PASS"})
    workspace.write_json("translation/en.json", {}, validate=False)
    workspace.write_json("translation/validation.json", {"status": "PASS"})
    workspace.write_json("subtitles/validation.json", {"status": "PASS"})
    for relative in (
        "subtitles/en.srt",
        "subtitles/en.vtt",
        "output/soft-subtitled.mp4",
        "output/burned-in.mp4",
    ):
        workspace.resolve(relative).write_text("synthetic", encoding="utf-8")
    workspace.write_json(
        "validation/report.json",
        {
            "quality_status": "PASS",
            "sampled_frames": [],
            "source_sampled_frames": [],
            "burned_lower_frame_difference_ratio": [],
        },
    )

    with pytest.raises(
        StageBlocked, match="Frame validation evidence is invalid"
    ) as captured:
        Pipeline(config, workspace).stage_manifest()

    assert captured.value.code == "VALIDATION_EVIDENCE_INVALID"

import json
import os
import subprocess
from pathlib import Path

from linguarelay.config import load_config
from linguarelay.models import Segment, Transcript, TranslationSegment
from linguarelay.pipeline import Pipeline, STAGES
from linguarelay.security import JobWorkspace
from linguarelay.security import sha256_file


class SyntheticASR:
    def __init__(self, name: str) -> None:
        self.name = name
        self.model = "synthetic-asr-v1"

    def transcribe(
        self, audio_path: Path, *, language: str, min_confidence: float
    ) -> Transcript:
        del min_confidence
        if audio_path.name == "clip-01.flac":
            segments = [
                Segment(id="one", start=0.0, end=1.0, text="Olá, Ana.", confidence=0.99)
            ]
        elif audio_path.name == "clip-02.flac":
            segments = [
                Segment(
                    id="two", start=0.0, end=1.0, text="O número é 42.", confidence=0.99
                )
            ]
        elif audio_path.name == "clip-03.flac":
            segments = [
                Segment(
                    id="three",
                    start=0.0,
                    end=1.0,
                    text="Fim do exemplo.",
                    confidence=0.99,
                )
            ]
        else:
            segments = [
                Segment(
                    id="one", start=0.2, end=1.2, text="Olá, Ana.", confidence=0.99
                ),
                Segment(
                    id="two", start=5.0, end=6.0, text="O número é 42.", confidence=0.99
                ),
                Segment(
                    id="three",
                    start=9.8,
                    end=10.8,
                    text="Fim do exemplo.",
                    confidence=0.99,
                ),
            ]
        return Transcript(
            language=language, provider=self.name, model=self.model, segments=segments
        )


class SyntheticTranslator:
    name = "synthetic-translation"
    model = "synthetic-translation-v1"

    def translate(self, segments, *, source_language, target_language, glossary_terms):
        del source_language, target_language, glossary_terms
        mapping = {
            "one": "Hello, Ana.",
            "two": "The number is 42.",
            "three": "End of the example.",
            "part1-one": "Hello, Ana.",
            "part1-two": "The number is 42.",
            "part1-three": "End of the example.",
        }
        return [
            TranslationSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                source_text=segment.text,
                text=mapping[segment.id],
            )
            for segment in segments
        ]


def _make_source(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=12",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=12",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )


def test_synthetic_pipeline_reaches_pass_manifest(tmp_path: Path) -> None:
    baseline = Path("config/example.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "synthetic.toml"
    config_path.write_text(
        baseline.replace(
            'jobs_root = "~/.local/share/linguarelay/jobs"',
            f'jobs_root = "{tmp_path / "external-jobs"}"',
        )
        .replace(
            "representative_clip_seconds = 45",
            "representative_clip_seconds = 2",
        )
        .replace(
            "representative_clip_positions = [0.12, 0.50, 0.82]",
            "representative_clip_positions = [0.10, 0.50, 0.90]",
        )
        ,
        encoding="utf-8",
    )
    config = load_config(config_path, repo_root=tmp_path / "repo")
    config.data["asr"]["candidates"] = ["fixture_a"]
    workspace = JobWorkspace.create(
        config.jobs_root, "synthetic-e2e", repo_root=config.repo_root
    )
    source = workspace.resolve("source/source.mp4")
    _make_source(source)
    os.chmod(source, 0o400)
    workspace.write_json(
        "source/discovery.json",
        {
            "schema_version": 1,
            "status": "PASS",
            "transport": "mp4",
            "source_path": "source/source.mp4",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "fixture": "synthetic",
        },
    )
    workspace.write_json(
        "asr/reference.json",
        {
            "schema_version": 1,
            "language": "pt-BR",
            "manually_reviewed": True,
            "reviewed_by": "synthetic-integration-test",
            "reviewed_at": "2026-07-18T00:00:00Z",
            "glossary_terms": ["Ana"],
            "numbers": ["42"],
            "segments": [
                {"id": "r1", "start": 0.2, "end": 1.2, "text": "Olá, Ana."},
                {"id": "r2", "start": 5.0, "end": 6.0, "text": "O número é 42."},
                {"id": "r3", "start": 9.8, "end": 10.8, "text": "Fim do exemplo."},
            ],
        },
        validate=False,
    )
    pipeline = Pipeline(
        config,
        workspace,
        asr_overrides={
            "fixture_a": SyntheticASR("fixture_a"),
        },
        translation_override=SyntheticTranslator(),
    )

    assert pipeline.run(stages=STAGES) == list(STAGES)

    manifest = json.loads(
        workspace.resolve("manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["quality_status"] == "PASS"
    assert workspace.resolve("transcript/pt-BR.raw.json").is_file()
    assert workspace.resolve("translation/en.json").is_file()
    assert workspace.resolve("subtitles/en.srt").is_file()
    assert workspace.resolve("output/soft-subtitled.mp4").is_file()
    assert workspace.resolve("output/burned-in.mp4").is_file()

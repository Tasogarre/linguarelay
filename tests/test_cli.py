import json
import stat
from pathlib import Path

import pytest

from linguarelay.cli import main


def _write_config(tmp_path: Path) -> Path:
    baseline = Path("config/example.toml").read_text(encoding="utf-8")
    config = tmp_path / "cli.toml"
    config.write_text(
        baseline.replace(
            'jobs_root = "~/.local/share/linguarelay/jobs"',
            f'jobs_root = "{tmp_path / "external-jobs"}"',
        ),
        encoding="utf-8",
    )
    return config


def test_cli_init_status_and_dry_run(tmp_path: Path, capsys) -> None:
    config = _write_config(tmp_path)

    assert main(["init", "--config", str(config), "--job-id", "cli-job"]) == 0
    assert main(["status", "--config", str(config), "--job-id", "cli-job"]) == 0
    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--job-id",
                "cli-job",
                "--dry-run",
                "--stages",
                "normalize,prepare-eval",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "cli-job" in output
    assert "prepare-eval" in output


def test_cli_ingest_copies_source_without_disclosing_original_path(
    tmp_path: Path, capsys
) -> None:
    config = _write_config(tmp_path)
    source = tmp_path / "private-name.mp4"
    source.write_bytes(b"synthetic-media")

    assert (
        main(
            [
                "ingest",
                "--config",
                str(config),
                "--job-id",
                "local-job",
                "--input",
                str(source),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert str(source) not in output
    job = tmp_path / "external-jobs/local-job"
    copied = job / "source/source.mp4"
    assert copied.read_bytes() == b"synthetic-media"
    assert stat.S_IMODE(copied.stat().st_mode) == 0o400
    receipt = json.loads((job / "source/discovery.json").read_text())
    assert receipt["transport"] == "local"
    assert receipt["source_path"] == "source/source.mp4"
    state = json.loads((job / "state.json").read_text())
    assert state["stages"]["acquire"]["status"] == "complete"


def test_cli_ingest_redacts_local_io_error_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = _write_config(tmp_path)
    source = tmp_path / "private-name.mp4"
    source.write_bytes(b"synthetic-media")

    def fail_ingest(self, _source):
        del self
        raise PermissionError(f"denied: {source}")

    monkeypatch.setattr(
        "linguarelay.cli.JobWorkspace.ingest_local_source", fail_ingest
    )

    assert (
        main(
            [
                "ingest",
                "--config",
                str(config),
                "--job-id",
                "local-error-job",
                "--input",
                str(source),
            ]
        )
        == 2
    )
    error = capsys.readouterr().err
    assert str(source) not in error
    assert "LOCAL_IO_FAILURE" in error


def test_cli_validate_refreshes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    config = _write_config(tmp_path)
    assert main(["init", "--config", str(config), "--job-id", "validate-job"]) == 0
    calls: list[list[str]] = []

    def fake_run(self, *, stages=None, resume=False, dry_run=False):
        del self, resume, dry_run
        calls.append(list(stages or []))
        return list(stages or [])

    monkeypatch.setattr("linguarelay.cli.Pipeline.run", fake_run)

    assert main(["validate", "--config", str(config), "--job-id", "validate-job"]) == 0
    assert calls == [["validate", "manifest"]]
    assert '"stages": ["validate", "manifest"]' in capsys.readouterr().out

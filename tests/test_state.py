from pathlib import Path

import pytest

from linguarelay.security import JobWorkspace, SensitiveDataError, sha256_file
from linguarelay.state import JobState


def test_resume_requires_matching_output_hashes(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "resume-job", repo_root=tmp_path / "repo"
    )
    output = workspace.resolve("audio/normalized.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"synthetic-audio")
    state = JobState(workspace)
    state.complete("normalize", [output])

    assert state.can_resume("normalize")
    assert state.data["status"] == "in_progress"
    assert state.data["stages"]["normalize"]["outputs"][0]["sha256"] == sha256_file(
        output
    )

    output.write_bytes(b"changed")
    assert not state.can_resume("normalize")


def test_manifest_completion_sets_overall_job_status(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "complete-job", repo_root=tmp_path / "repo"
    )
    manifest = workspace.write_json("manifest.json", {"quality_status": "PASS"})
    state = JobState(workspace)

    state.complete("manifest", [manifest])

    assert state.data["status"] == "complete"


def test_invalid_state_is_rejected_without_echoing_contents(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "bad-state", repo_root=tmp_path / "repo"
    )
    workspace.resolve("state.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(SensitiveDataError, match="job state is invalid"):
        JobState(workspace)

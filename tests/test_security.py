import json
import stat
from pathlib import Path

import pytest

from linguarelay.security import (
    SensitiveDataError,
    JobWorkspace,
    redact_text,
    validate_redacted_mapping,
)


def test_redactor_removes_signed_query_and_auth_material() -> None:
    raw = (
        "GET https://cdn.example/video.m3u8?token=abc&expires=123 "
        "Authorization: Bearer secret Cookie: sid=value"
    )
    redacted = redact_text(raw)

    assert "abc" not in redacted
    assert "secret" not in redacted
    assert "sid=value" not in redacted
    assert "[REDACTED_URL]" in redacted

    assert "basic-secret" not in redact_text("Authorization: Basic basic-secret")


def test_manifest_schema_refuses_sensitive_keys() -> None:
    for key in ("signed_url", "api_key", "access-key", "private_key"):
        with pytest.raises(SensitiveDataError):
            validate_redacted_mapping({"stage": "acquire", key: "not-safe"})

    validate_redacted_mapping(
        {"stage": "acquire", "transport": "hls", "sha256": "a" * 64}
    )

    with pytest.raises(SensitiveDataError):
        validate_redacted_mapping(
            {"stage": "acquire", "note": "https://cdn.invalid/media/path"}
        )


def test_workspace_is_private_atomic_and_confined(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "job-001", repo_root=tmp_path / "repo"
    )
    workspace.write_json("state.json", {"schema_version": 1, "status": "running"})

    assert stat.S_IMODE(workspace.root.stat().st_mode) == 0o700
    assert (
        json.loads((workspace.root / "state.json").read_text())["status"] == "running"
    )
    assert not list(workspace.root.glob("*.tmp"))

    with pytest.raises(SensitiveDataError):
        workspace.resolve("../escape")


def test_open_rejects_group_readable_workspace(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    workspace = JobWorkspace.create(jobs, "job-unsafe", repo_root=tmp_path / "repo")
    workspace.root.chmod(0o750)

    with pytest.raises(SensitiveDataError, match="owner-only"):
        JobWorkspace.open(jobs, "job-unsafe")


def test_private_directory_rejects_extended_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    monkeypatch.setattr(JobWorkspace, "_has_acl", staticmethod(lambda _path: True))

    with pytest.raises(SensitiveDataError, match="owner-only"):
        JobWorkspace._require_private_directory(private, "private directory")


def test_create_rejects_broad_or_shared_jobs_root(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(SensitiveDataError, match="jobs_root must be owner-only"):
        JobWorkspace.create(shared, "job-unsafe", repo_root=tmp_path / "repo")

    with pytest.raises(SensitiveDataError, match="too broad"):
        JobWorkspace.create(Path.home(), "job-unsafe", repo_root=tmp_path / "repo")


def test_open_rejects_symlinked_job_workspace(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    jobs.mkdir(mode=0o700)
    jobs.chmod(0o700)
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    external.chmod(0o700)
    (jobs / "linked-job").symlink_to(external, target_is_directory=True)

    with pytest.raises(SensitiveDataError, match="symlink"):
        JobWorkspace.open(jobs, "linked-job")


def test_local_ingest_rejects_symlinks_and_duplicate_sources(tmp_path: Path) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "ingest-job", repo_root=tmp_path / "repo"
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    linked = tmp_path / "linked.mp4"
    linked.symlink_to(source)

    with pytest.raises(SensitiveDataError, match="non-symlink"):
        workspace.ingest_local_source(linked)

    workspace.ingest_local_source(source)
    assert workspace.ingest_local_source(source).name == "source.mp4"

    other = tmp_path / "other.mp4"
    other.write_bytes(b"different")
    with pytest.raises(SensitiveDataError, match="already contains"):
        workspace.ingest_local_source(other)


def test_local_ingest_removes_partial_copy_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = JobWorkspace.create(
        tmp_path / "jobs", "failed-ingest", repo_root=tmp_path / "repo"
    )
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    def fail_copy(input_handle, output_handle):
        del input_handle
        output_handle.write(b"partial")
        raise OSError("synthetic copy failure")

    monkeypatch.setattr("linguarelay.security._copy_and_hash", fail_copy)
    with pytest.raises(OSError, match="synthetic copy failure"):
        workspace.ingest_local_source(source)

    assert not list(workspace.resolve("source").iterdir())
    monkeypatch.undo()
    assert workspace.ingest_local_source(source).read_bytes() == b"synthetic"

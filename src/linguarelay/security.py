from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


class SensitiveDataError(ValueError):
    """Raised before sensitive material can be persisted or escape a job root."""


_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_AUTH = re.compile(r"(?i)(authorization\s*:\s*)[^\r\n]+?(?=\s+[A-Za-z-]+\s*:|$)")
_COOKIE = re.compile(r"(?i)(cookie\s*:\s*)[^\r\n]+")
_TOKEN_PAIR = re.compile(
    r"(?i)\b(token|signature|sig|key|secret|password|otp)=([^&\s]+)"
)
_SENSITIVE_KEY_PARTS = {
    "authorization",
    "access_key",
    "api_key",
    "auth_header",
    "browser_profile",
    "cookie",
    "credential",
    "email",
    "har",
    "header",
    "media_url",
    "otp",
    "password",
    "private_key",
    "profile_data",
    "request_url",
    "secret",
    "signed_url",
    "storage_state",
    "token",
    "uri",
    "url",
}
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def redact_text(value: str) -> str:
    redacted = _URL.sub("[REDACTED_URL]", value)
    redacted = _AUTH.sub(r"\1[REDACTED]", redacted)
    redacted = _COOKIE.sub(r"\1[REDACTED]", redacted)
    redacted = _TOKEN_PAIR.sub(r"\1=[REDACTED]", redacted)
    return redacted


def _sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    return any(
        normalized == part
        or normalized.startswith(f"{part}_")
        or normalized.endswith(f"_{part}")
        for part in _SENSITIVE_KEY_PARTS
    )


def validate_redacted_mapping(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _sensitive_key(str(key)):
                raise SensitiveDataError(
                    "sensitive fields are forbidden in persisted metadata"
                )
            validate_redacted_mapping(nested)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            validate_redacted_mapping(nested)
        return
    if isinstance(value, str):
        if redact_text(value) != value:
            raise SensitiveDataError(
                "sensitive-looking values are forbidden in persisted metadata"
            )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_and_hash(input_handle: Any, output_handle: Any) -> str:
    digest = hashlib.sha256()
    while chunk := input_handle.read(1024 * 1024):
        output_handle.write(chunk)
        digest.update(chunk)
    return digest.hexdigest()


class JobWorkspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def create(cls, jobs_root: Path, job_id: str, *, repo_root: Path) -> "JobWorkspace":
        if not _JOB_ID.fullmatch(job_id):
            raise SensitiveDataError("job_id must use only safe opaque characters")
        root_parent = jobs_root.expanduser().resolve()
        repository = repo_root.resolve()
        home = Path.home().resolve()
        if root_parent in {Path(root_parent.anchor), home, home.parent}:
            raise SensitiveDataError("jobs_root is too broad for private job storage")
        try:
            root_parent.relative_to(repository)
        except ValueError:
            pass
        else:
            raise SensitiveDataError("job storage must be outside the repository")
        try:
            repository.relative_to(root_parent)
        except ValueError:
            pass
        else:
            raise SensitiveDataError("jobs_root must not contain the repository")
        if root_parent.exists():
            cls._strip_acl(root_parent)
            cls._require_private_directory(root_parent, "jobs_root")
        else:
            root_parent.mkdir(parents=True, mode=0o700)
            os.chmod(root_parent, 0o700)
            cls._strip_acl(root_parent)
        root = root_parent / job_id
        if root.is_symlink():
            raise SensitiveDataError("job workspace must not be a symlink")
        if root.exists():
            cls._strip_acl(root)
            cls._require_private_directory(root, "job workspace")
        else:
            root.mkdir(mode=0o700)
            os.chmod(root, 0o700)
            cls._strip_acl(root)
        for name in (
            "source",
            "audio",
            "asr",
            "transcript",
            "translation",
            "subtitles",
            "output",
            "validation",
            "logs",
        ):
            directory = root / name
            if directory.is_symlink():
                raise SensitiveDataError("job subdirectories must not be symlinks")
            if directory.exists():
                cls._strip_acl(directory)
                cls._require_private_directory(directory, "job subdirectory")
            else:
                directory.mkdir(mode=0o700)
                os.chmod(directory, 0o700)
                cls._strip_acl(directory)
        return cls(root)

    @classmethod
    def open(cls, jobs_root: Path, job_id: str) -> "JobWorkspace":
        if not _JOB_ID.fullmatch(job_id):
            raise SensitiveDataError("invalid job_id")
        root_parent = jobs_root.expanduser().resolve()
        candidate = root_parent / job_id
        if candidate.is_symlink():
            raise SensitiveDataError("job workspace must not be a symlink")
        root = candidate.resolve()
        try:
            root.relative_to(root_parent)
        except ValueError:
            raise SensitiveDataError(
                "job workspace escaped the configured jobs root"
            ) from None
        if not root.is_dir():
            raise FileNotFoundError("job does not exist")
        cls._require_private_directory(root, "job workspace")
        return cls(root)

    @staticmethod
    def _require_private_directory(path: Path, label: str) -> None:
        if not path.is_dir():
            raise SensitiveDataError(f"{label} must be a directory")
        details = path.stat()
        if (
            details.st_uid != os.getuid()
            or details.st_mode & 0o077
            or JobWorkspace._has_acl(path)
        ):
            raise SensitiveDataError(f"{label} must be owner-only")

    @staticmethod
    def _strip_acl(path: Path) -> None:
        if sys.platform != "darwin":
            return
        result = subprocess.run(
            ["chmod", "-N", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise SensitiveDataError("unable to enforce owner-only directory access")

    @staticmethod
    def _has_acl(path: Path) -> bool:
        if sys.platform != "darwin":
            return False
        result = subprocess.run(
            ["ls", "-lde", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return True
        mode = result.stdout.split(maxsplit=1)[0] if result.stdout else ""
        return mode.endswith("+")

    def resolve(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise SensitiveDataError("job path escaped the workspace") from None
        return candidate

    def ingest_local_source(self, source: Path) -> Path:
        """Copy one user-selected local media file into the private job."""
        source = source.expanduser()
        try:
            source_before = source.lstat()
        except OSError:
            raise SensitiveDataError("local source is unavailable") from None
        if source.is_symlink() or not stat.S_ISREG(source_before.st_mode):
            raise SensitiveDataError("local source must be a regular, non-symlink file")
        if source_before.st_size <= 0:
            raise SensitiveDataError("local source must not be empty")
        suffix = source.suffix.casefold()
        if suffix not in {".mp4", ".mkv", ".mov", ".webm", ".ts", ".m4v"}:
            raise SensitiveDataError("unsupported local media extension")
        destination = self.resolve(f"source/source{suffix}")
        try:
            source_descriptor = os.open(
                source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            )
        except OSError:
            raise SensitiveDataError("local source could not be opened safely") from None
        with os.fdopen(source_descriptor, "rb") as source_handle:
            source_opened = os.fstat(source_handle.fileno())
            if (
                not stat.S_ISREG(source_opened.st_mode)
                or (source_before.st_dev, source_before.st_ino)
                != (source_opened.st_dev, source_opened.st_ino)
            ):
                raise SensitiveDataError("local source changed during validation")

            digest = hashlib.sha256()
            while chunk := source_handle.read(1024 * 1024):
                digest.update(chunk)
            source_sha256 = digest.hexdigest()
            source_handle.seek(0)
            existing = [
                path
                for path in self.resolve("source").glob("source.*")
                if path.is_file()
            ]
            if (
                len(existing) == 1
                and existing[0] == destination
                and sha256_file(existing[0]) == source_sha256
            ):
                return existing[0]
            if existing:
                raise SensitiveDataError("job already contains a source media file")
            descriptor, temporary = tempfile.mkstemp(
                prefix=".source.", suffix=".partial", dir=destination.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as output_handle:
                    copied_sha256 = _copy_and_hash(source_handle, output_handle)
                    output_handle.flush()
                    os.fsync(output_handle.fileno())
                if copied_sha256 != source_sha256:
                    raise SensitiveDataError("local source changed while it was copied")
                os.chmod(temporary, 0o400)
                os.replace(temporary, destination)
            except BaseException:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise
        return destination

    def write_json(
        self, relative: str | Path, value: Any, *, validate: bool = True
    ) -> Path:
        if validate:
            validate_redacted_mapping(value)
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            os.chmod(target, 0o600)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return target

    def append_event(self, stage: str, status: str, *, code: str | None = None) -> Path:
        event = {"stage": stage, "status": status}
        if code:
            event["code"] = code
        validate_redacted_mapping(event)
        target = self.resolve("logs/pipeline.jsonl")
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        os.chmod(target, 0o600)
        return target

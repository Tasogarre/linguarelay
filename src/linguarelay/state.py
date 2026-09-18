from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .security import JobWorkspace, SensitiveDataError, sha256_file


class JobState:
    def __init__(self, workspace: JobWorkspace) -> None:
        self.workspace = workspace
        state_path = workspace.resolve("state.json")
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raise SensitiveDataError("job state is invalid") from None
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or not isinstance(payload.get("stages"), dict)
            ):
                raise SensitiveDataError("job state is invalid")
            self.data: dict[str, Any] = payload
        else:
            self.data = {"schema_version": 1, "status": "initialized", "stages": {}}

    def save(self) -> None:
        self.workspace.write_json("state.json", self.data)

    def start(self, stage: str) -> None:
        self.data["status"] = "running"
        self.data.setdefault("stages", {})[stage] = {
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
            "outputs": [],
        }
        self.save()
        self.workspace.append_event(stage, "running")

    def complete(
        self,
        stage: str,
        outputs: Iterable[Path],
        *,
        context_sha256: str | None = None,
    ) -> None:
        records = []
        for output in outputs:
            resolved = output.resolve()
            relative = resolved.relative_to(self.workspace.root)
            records.append(
                {
                    "path": relative.as_posix(),
                    "bytes": resolved.stat().st_size,
                    "sha256": sha256_file(resolved),
                }
            )
        record = {
            "status": "complete",
            "completed_at": datetime.now(UTC).isoformat(),
            "outputs": records,
        }
        if context_sha256:
            record["context_sha256"] = context_sha256
        self.data.setdefault("stages", {})[stage] = record
        self.data["status"] = "complete" if stage == "manifest" else "in_progress"
        self.save()
        self.workspace.append_event(stage, "complete")

    def fail(self, stage: str, code: str) -> None:
        self.data["status"] = "blocked"
        self.data.setdefault("stages", {})[stage] = {
            "status": "blocked",
            "code": code,
            "failed_at": datetime.now(UTC).isoformat(),
            "outputs": [],
        }
        self.save()
        self.workspace.append_event(stage, "blocked", code=code)

    def can_resume(self, stage: str, *, context_sha256: str | None = None) -> bool:
        record = self.data.get("stages", {}).get(stage, {})
        outputs = record.get("outputs")
        if (
            record.get("status") != "complete"
            or not isinstance(outputs, list)
            or not outputs
        ):
            return False
        if (
            context_sha256 is not None
            and record.get("context_sha256") != context_sha256
        ):
            return False
        for output in outputs:
            if not isinstance(output, dict) or not isinstance(output.get("path"), str):
                return False
            path = self.workspace.resolve(output["path"])
            if not path.is_file() or sha256_file(path) != output.get("sha256"):
                return False
        return True

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, load_config
from .pipeline import ALL_STAGES, Pipeline, StageBlocked
from .security import JobWorkspace, SensitiveDataError, sha256_file
from .state import JobState


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--job-id", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linguarelay")
    subcommands = parser.add_subparsers(dest="command", required=True)

    initialize = subcommands.add_parser(
        "init", help="create a private external job workspace"
    )
    _common(initialize)

    ingest = subcommands.add_parser("ingest", help="copy local media into a private job")
    _common(ingest)
    ingest.add_argument("--input", type=Path, required=True)

    status = subcommands.add_parser("status", help="show redacted stage state")
    _common(status)

    run = subcommands.add_parser("run", help="run an ordered set of stages")
    _common(run)
    run.add_argument("--stages", help="comma-separated explicit stages")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    stage = subcommands.add_parser("stage", help="run one explicit stage")
    _common(stage)
    stage.add_argument("stage", choices=ALL_STAGES)
    stage.add_argument("--resume", action="store_true")
    stage.add_argument("--dry-run", action="store_true")

    validate = subcommands.add_parser(
        "validate", help="validate outputs and refresh the final manifest"
    )
    _common(validate)
    validate.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config, repo_root=Path.cwd())
        if args.command == "ingest":
            workspace = JobWorkspace.create(
                config.jobs_root, args.job_id, repo_root=config.repo_root
            )
            source = workspace.ingest_local_source(args.input)
            receipt = workspace.write_json(
                "source/discovery.json",
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "transport": "local",
                    "source_path": source.relative_to(workspace.root).as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                },
            )
            state = JobState(workspace)
            state.complete("acquire", [source, receipt])
            print(
                json.dumps(
                    {
                        "status": "ingested",
                        "job_id": args.job_id,
                        "source": source.relative_to(workspace.root).as_posix(),
                    }
                )
            )
            return 0
        if args.command == "init":
            workspace = JobWorkspace.create(
                config.jobs_root, args.job_id, repo_root=config.repo_root
            )
            state = JobState(workspace)
            state.save()
            print(
                json.dumps(
                    {
                        "status": "initialized",
                        "job_id": args.job_id,
                    }
                )
            )
            return 0

        workspace = JobWorkspace.open(config.jobs_root, args.job_id)
        if args.command == "status":
            state = JobState(workspace)
            print(json.dumps(state.data, indent=2, sort_keys=True))
            return 0

        pipeline = Pipeline(config, workspace)
        if args.command == "run":
            stages = (
                [item.strip() for item in args.stages.split(",") if item.strip()]
                if args.stages
                else None
            )
            executed = pipeline.run(
                stages=stages, resume=args.resume, dry_run=args.dry_run
            )
        elif args.command == "stage":
            executed = pipeline.run(
                stages=[args.stage], resume=args.resume, dry_run=args.dry_run
            )
        else:
            executed = pipeline.run(stages=["validate", "manifest"], resume=args.resume)
        print(
            json.dumps(
                {
                    "status": "dry-run" if getattr(args, "dry_run", False) else "ok",
                    "job_id": args.job_id,
                    "stages": executed,
                }
            )
        )
        return 0
    except StageBlocked as exc:
        print(
            json.dumps(
                {"status": "blocked", "code": exc.code, "message": exc.public_message}
            ),
            file=sys.stderr,
        )
        return 2
    except (ConfigError, SensitiveDataError, RuntimeError) as exc:
        print(
            json.dumps(
                {"status": "error", "code": type(exc).__name__, "message": str(exc)}
            ),
            file=sys.stderr,
        )
        return 2
    except OSError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "code": "LOCAL_IO_FAILURE",
                    "message": "A local file operation failed.",
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

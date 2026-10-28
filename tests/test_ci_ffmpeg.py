"""CI readiness is partial feedback; full media assertions remain in the suite."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check-ci-ffmpeg.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("check_ci_ffmpeg", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventories():
    return {
        "-encoders": "\n".join(
            f" V..... {name} synthetic entry"
            for name in "libx264 aac pcm_s16le flac qtrle png mov_text srt".split()
        ),
        "-decoders": "\n".join(
            f" V..... {name} synthetic entry"
            for name in "h264 aac pcm_s16le flac qtrle png subrip".split()
        ),
        "-filters": "\n".join(
            f" .. {name} synthetic entry"
            for name in "overlay format testsrc2 color sine".split()
        ),
    }


def _probe():
    return {
        "streams": [
            {"codec_type": "video", "codec_name": "h264", "width": 64, "height": 64},
            {"codec_type": "audio", "codec_name": "aac"},
        ],
        "format": {"duration": "0.5"},
    }


def _runner(*, inventories=None, probe=None, fail_stage=None):
    inventories = _inventories() if inventories is None else inventories
    probe = _probe() if probe is None else probe

    def run(command):
        for flag, listing in inventories.items():
            if flag in command:
                return listing
        stage = "probe" if command[0] == "ffprobe" else "decode" if "null" in command else "encode"
        if stage == fail_stage:
            raise subprocess.CalledProcessError(1, command)
        return json.dumps(probe) if stage == "probe" else ""

    return run


@pytest.mark.parametrize("filter_flags", ["..", "..."])
def test_complete_capability_proof_is_labelled_partial(checker, capsys, filter_flags):
    inventories = _inventories()
    inventories["-filters"] = inventories["-filters"].replace(" .. ", f" {filter_flags} ")
    with patch.object(checker, "run", side_effect=_runner(inventories=inventories)):
        assert checker.main() == 0
    assert "partial" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-encoders", "-decoders", "-filters"])
def test_each_required_capability_must_be_present(checker, flag):
    inventories = _inventories()
    for line in inventories[flag].splitlines():
        incomplete = dict(inventories)
        incomplete[flag] = inventories[flag].replace(line, "")
        with patch.object(checker, "run", side_effect=_runner(inventories=incomplete)):
            assert checker.main() == 1, line


def test_capability_names_require_exact_tokens(checker):
    inventories = _inventories()
    inventories["-encoders"] = inventories["-encoders"].replace(" qtrle ", " qtrle_fake ")
    with patch.object(checker, "run", side_effect=_runner(inventories=inventories)):
        assert checker.main() == 1


@pytest.mark.parametrize("stage", ["encode", "probe", "decode"])
def test_synthetic_command_failure_requires_installation(checker, stage):
    with patch.object(checker, "run", side_effect=_runner(fail_stage=stage)):
        assert checker.main() == 1


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("ffmpeg", 20)])
def test_missing_or_timed_out_executables_require_installation(checker, error):
    with patch.object(checker, "run", side_effect=error):
        assert checker.main() == 1


@pytest.mark.parametrize("payload", ["{", "[]", '{"streams": [], "format": {"duration": "0.5"}}'])
def test_unproven_probe_output_requires_installation(checker, payload):
    runner = _runner()
    with patch.object(checker, "run", side_effect=lambda command: payload if command[0] == "ffprobe" else runner(command)):
        assert checker.main() == 1


@pytest.mark.parametrize(
    "stream_type,field,value",
    [
        ("video", "codec_name", "hevc"),
        ("audio", "codec_name", "mp3"),
        ("video", "width", 32),
        ("video", "height", 32),
    ],
)
def test_probe_must_confirm_expected_encoded_streams(checker, stream_type, field, value):
    probe = _probe()
    for stream in probe["streams"]:
        if stream["codec_type"] == stream_type:
            stream[field] = value
    with patch.object(checker, "run", side_effect=_runner(probe=probe)):
        assert checker.main() == 1


@pytest.mark.parametrize("duration", ["0", "nan", "inf", "unavailable"])
def test_invalid_synthetic_duration_requires_installation(checker, duration):
    probe = _probe()
    probe["format"]["duration"] = duration
    with patch.object(checker, "run", side_effect=_runner(probe=probe)):
        assert checker.main() == 1


def _provisioning_shell():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    step = workflow.split("      - name: Ensure FFmpeg capabilities\n", 1)[1]
    script = step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    return "\n".join(line[10:] for line in script.splitlines())


@pytest.mark.parametrize(
    "first_probe,second_probe,apt_exit,expected_exit,expected_calls",
    [
        (0, 0, 0, 0, ["probe"]),
        (1, 0, 0, 0, ["probe", "apt-get update", "apt-get install --yes ffmpeg", "probe"]),
        (1, 1, 0, 1, ["probe", "apt-get update", "apt-get install --yes ffmpeg", "probe"]),
        (1, 0, 1, 1, ["probe", "apt-get update"]),
    ],
)
def test_actual_ci_shell_retains_fallback_and_requires_success(
    tmp_path, first_probe, second_probe, apt_exit, expected_exit, expected_calls
):
    (tmp_path / "uv").write_text(
        '#!/bin/sh\necho probe >> "$CALLS"\n'
        'if test -f "$MARKER"; then exit "$SECOND_PROBE"; fi\n'
        'touch "$MARKER"\nexit "$FIRST_PROBE"\n'
    )
    (tmp_path / "sudo").write_text(
        '#!/bin/sh\necho "$*" >> "$CALLS"\nexit "$APT_EXIT"\n'
    )
    for name in ("uv", "sudo"):
        (tmp_path / name).chmod(0o700)
    calls = tmp_path / "calls"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.defpath}",
        "CALLS": str(calls),
        "MARKER": str(tmp_path / "marker"),
        "FIRST_PROBE": str(first_probe),
        "SECOND_PROBE": str(second_probe),
        "APT_EXIT": str(apt_exit),
    }
    result = subprocess.run(["/bin/bash", "-e", "-c", _provisioning_shell()], env=env, check=False)
    assert result.returncode == expected_exit
    assert calls.read_text().splitlines() == expected_calls


def test_full_tests_and_distribution_build_remain_unconditional():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    for step, command in [("Run tests", "uv run pytest -q"), ("Build distributions", "uv build --no-sources")]:
        block = workflow.split(f"      - name: {step}\n", 1)[1].split("\n      - name:", 1)[0]
        assert f"run: {command}" in block
        assert "if:" not in block

#!/usr/bin/env python3
"""Prove runner readiness using only short-lived synthetic media, never full proof."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path


# Match the default media pipeline and its synthetic integration fixtures.
REQUIRED = {
    "-encoders": {"libx264", "aac", "pcm_s16le", "flac", "qtrle", "png", "mov_text", "srt"},
    "-decoders": {"h264", "aac", "pcm_s16le", "flac", "qtrle", "png", "subrip"},
    "-filters": {"overlay", "format", "testsrc2", "color", "sine"},
}


def run(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=20,
    ).stdout


def check_capabilities() -> None:
    for flag, required in REQUIRED.items():
        listing = run(["ffmpeg", "-hide_banner", flag])
        available = set(re.findall(r"^\s*[A-Z.]{2,6}\s+(\w+)\s", listing, re.MULTILINE))
        missing = required - available
        if missing:
            raise ValueError(f"missing {flag}: {', '.join(sorted(missing))}")

    with tempfile.TemporaryDirectory(prefix="linguarelay-ci-ffmpeg-") as directory:
        output = Path(directory) / "synthetic.mp4"
        run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=4:duration=0.5",
            "-f", "lavfi", "-i", "color=c=white@0.5:size=64x64:rate=4:duration=0.5,format=rgba",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=0.5",
            "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto:shortest=1,format=yuv420p[v]",
            "-map", "[v]", "-map", "2:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-t", "0.5", str(output),
        ])
        probe = json.loads(run([
            "ffprobe", "-v", "error", "-show_streams", "-show_format",
            "-of", "json", str(output),
        ]))
        streams = probe["streams"]
        video = any(
            stream.get("codec_type") == "video"
            and stream.get("codec_name") == "h264"
            and stream.get("width") == 64
            and stream.get("height") == 64
            for stream in streams
        )
        audio = any(
            stream.get("codec_type") == "audio" and stream.get("codec_name") == "aac"
            for stream in streams
        )
        duration = float(probe["format"]["duration"])
        if not (video and audio and math.isfinite(duration) and duration > 0):
            raise ValueError("synthetic output did not prove video, audio and duration")
        run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(output), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-",
        ])


def main() -> int:
    try:
        check_capabilities()
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(f"FFmpeg capability probe failed ({type(error).__name__}); provisioning is required.")
        return 1
    print("FFmpeg capability probe passed (partial runner readiness; full tests and build still required).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

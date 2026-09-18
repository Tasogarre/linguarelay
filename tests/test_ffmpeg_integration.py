import shutil
import subprocess
from pathlib import Path

import pytest

from linguarelay.config import SubtitlePolicy
from linguarelay.ffmpeg import (
    burn_subtitles,
    decode_smoke,
    lower_frame_difference_ratio,
    mux_soft_subtitles,
    normalize_audio,
    probe_media,
    sample_cue_frames,
)
from linguarelay.models import Cue
from linguarelay.subtitles import render_srt


pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg is required",
)


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
            "testsrc2=size=320x180:rate=24:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=4",
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


def test_media_pipeline_creates_audio_soft_burned_and_validation_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    audio = tmp_path / "normalized.wav"
    srt_path = tmp_path / "en.srt"
    soft = tmp_path / "soft.mp4"
    soft_mkv = tmp_path / "soft.mkv"
    burned = tmp_path / "burned.mp4"
    _make_source(source)

    cues = [Cue(index=1, start=0.5, end=2.8, text="Synthetic subtitle")]
    srt_path.write_text(render_srt(cues), encoding="utf-8")
    policy = SubtitlePolicy(
        max_chars_per_line=42,
        max_lines=2,
        max_chars_per_second=20.0,
        min_cue_seconds=0.8,
        max_cue_seconds=7.0,
        minimum_gap_seconds=0.05,
        font_path="",
        font_size_ratio=0.06,
        bottom_margin_ratio=0.08,
    )

    normalize_audio(source, audio)
    mux_soft_subtitles(source, srt_path, soft)
    mux_soft_subtitles(source, srt_path, soft_mkv)
    burn_subtitles(source, cues, burned, policy, work_dir=tmp_path / "overlay")
    frames = sample_cue_frames(burned, cues, tmp_path / "frames", count=1)
    source_frames = sample_cue_frames(source, cues, tmp_path / "source-frames", count=1)

    soft_probe = probe_media(soft)
    soft_mkv_probe = probe_media(soft_mkv)
    burned_probe = probe_media(burned)
    assert audio.exists() and audio.stat().st_size > 0
    assert any(stream["codec_type"] == "subtitle" for stream in soft_probe["streams"])
    assert any(
        stream["codec_type"] == "subtitle" for stream in soft_mkv_probe["streams"]
    )
    assert any(stream["codec_type"] == "video" for stream in burned_probe["streams"])
    assert decode_smoke(soft, seconds=1.0)
    assert decode_smoke(soft_mkv, seconds=1.0)
    assert decode_smoke(burned, seconds=1.0)
    assert len(frames) == 1 and frames[0].stat().st_size > 0
    assert lower_frame_difference_ratio(source_frames[0], frames[0]) >= 0.002

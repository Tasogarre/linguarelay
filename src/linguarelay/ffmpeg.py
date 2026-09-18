from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageChops, ImageDraw, ImageFont

from .config import SubtitlePolicy
from .models import Cue


class MediaError(RuntimeError):
    """A media subprocess failed without leaking its full arguments or stderr."""


def _run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise MediaError(f"media command failed with exit code {result.returncode}")


def probe_media(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise MediaError("ffprobe failed")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise MediaError("ffprobe returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise MediaError("ffprobe returned an unexpected shape")
    return payload


def _duration_from_probe(probe: dict[str, Any]) -> float:
    try:
        duration = float(probe["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        durations = [
            float(stream["duration"])
            for stream in probe.get("streams", [])
            if stream.get("duration")
        ]
        if not durations:
            raise MediaError("media duration is unavailable") from None
        duration = max(durations)
    if not math.isfinite(duration) or duration <= 0:
        raise MediaError("media duration is invalid")
    return duration


def media_duration(path: Path, *, ffprobe: str = "ffprobe") -> float:
    return _duration_from_probe(probe_media(path, ffprobe=ffprobe))


def normalize_audio(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg: str = "ffmpeg",
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
    )
    return destination


def extract_audio_flac(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Extract a mono 16 kHz FLAC audio track from a media file.

    FLAC is lossless, so the extracted audio is a faithful, deterministic input
    for language detection and transcription. Only the first audio stream is used.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "flac",
            str(destination),
        ]
    )
    return destination


def extract_clip(
    source: Path,
    destination: Path,
    *,
    start: float,
    duration: float,
    ffmpeg: str = "ffmpeg",
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{max(0.0, start):.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:a",
            "flac",
            str(destination),
        ]
    )
    return destination


def split_audio(
    source: Path, destination_pattern: Path, *, seconds: int, ffmpeg: str = "ffmpeg"
) -> list[Path]:
    destination_pattern.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-f",
            "segment",
            "-segment_time",
            str(seconds),
            "-reset_timestamps",
            "1",
            "-c:a",
            "flac",
            str(destination_pattern),
        ]
    )
    return sorted(
        destination_pattern.parent.glob(destination_pattern.name.replace("%03d", "*"))
    )


def mux_soft_subtitles(
    source: Path, subtitles: Path, destination: Path, *, ffmpeg: str = "ffmpeg"
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    is_mp4 = destination.suffix.casefold() == ".mp4"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(subtitles),
        "-map",
        "0:v?",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        "mov_text" if is_mp4 else "srt",
        "-metadata:s:s:0",
        "language=eng",
        "-disposition:s:0",
        "default",
    ]
    if is_mp4:
        command.extend(["-movflags", "+faststart"])
    command.append(str(destination))
    _run(command)
    return destination


def _video_geometry(source: Path, *, ffprobe: str = "ffprobe") -> tuple[int, int]:
    probe = probe_media(source, ffprobe=ffprobe)
    for stream in probe.get("streams", []):
        if (
            stream.get("codec_type") == "video"
            and stream.get("width")
            and stream.get("height")
        ):
            return int(stream["width"]), int(stream["height"])
    raise MediaError("source has no video geometry")


def _draw_cue(
    path: Path, cue: Cue, width: int, height: int, policy: SubtitlePolicy
) -> None:
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font_size = max(18, round(height * policy.font_size_ratio))
    try:
        font = (
            ImageFont.truetype(policy.font_path, font_size)
            if policy.font_path
            else ImageFont.load_default(size=font_size)
        )
    except OSError:
        raise MediaError("configured subtitle font is unavailable") from None
    lines = cue.text.splitlines() or [cue.text]
    spacing = max(3, font_size // 6)
    boxes = [
        draw.textbbox((0, 0), line, font=font, stroke_width=max(1, font_size // 18))
        for line in lines
    ]
    text_width = max(box[2] - box[0] for box in boxes)
    line_heights = [box[3] - box[1] for box in boxes]
    text_height = sum(line_heights) + spacing * max(0, len(lines) - 1)
    padding_x = max(10, font_size // 2)
    padding_y = max(6, font_size // 4)
    x = (width - text_width) // 2
    bottom_margin = round(height * policy.bottom_margin_ratio)
    y = height - bottom_margin - text_height
    draw.rounded_rectangle(
        (
            x - padding_x,
            y - padding_y,
            x + text_width + padding_x,
            y + text_height + padding_y,
        ),
        radius=max(5, font_size // 5),
        fill=(0, 0, 0, 190),
    )
    cursor = y
    for line, line_height in zip(lines, line_heights, strict=True):
        line_box = draw.textbbox(
            (0, 0), line, font=font, stroke_width=max(1, font_size // 18)
        )
        line_width = line_box[2] - line_box[0]
        draw.text(
            ((width - line_width) // 2, cursor),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=max(1, font_size // 18),
            stroke_fill=(0, 0, 0, 255),
        )
        cursor += line_height + spacing
    image.save(path)


def _concat_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def burn_subtitles(
    source: Path,
    cues: Iterable[Cue],
    destination: Path,
    policy: SubtitlePolicy,
    *,
    work_dir: Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    video_codec: str = "libx264",
    audio_codec: str = "copy",
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    cue_list = list(cues)
    if not cue_list:
        raise MediaError("cannot burn an empty subtitle track")
    work_dir.mkdir(parents=True, exist_ok=True)
    width, height = _video_geometry(source, ffprobe=ffprobe)
    duration = media_duration(source, ffprobe=ffprobe)
    transparent = work_dir / "transparent.png"
    Image.new("RGBA", (width, height), (0, 0, 0, 0)).save(transparent)

    timeline: list[tuple[Path, float]] = []
    cursor = 0.0
    for index, cue in enumerate(cue_list, start=1):
        if cue.start > cursor:
            timeline.append((transparent, cue.start - cursor))
        cue_path = work_dir / f"cue-{index:06d}.png"
        _draw_cue(cue_path, cue, width, height, policy)
        timeline.append((cue_path, max(0.001, cue.end - cue.start)))
        cursor = cue.end
    if cursor < duration:
        timeline.append((transparent, duration - cursor))

    concat_path = work_dir / "overlay.ffconcat"
    lines = ["ffconcat version 1.0"]
    for image_path, item_duration in timeline:
        lines.append(f"file '{_concat_escape(image_path)}'")
        lines.append(f"duration {item_duration:.6f}")
    lines.append(f"file '{_concat_escape(timeline[-1][0])}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    overlay_path = work_dir / "overlay.mov"
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-fps_mode",
            "vfr",
            "-vf",
            "format=rgba",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "qtrle",
            str(overlay_path),
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-i",
        str(overlay_path),
        "-filter_complex",
        "[0:v][1:v]overlay=0:0:format=auto:shortest=1[v]",
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        audio_codec,
        "-movflags",
        "+faststart",
        "-t",
        f"{duration:.6f}",
        str(destination),
    ]
    try:
        _run(output_command)
    except MediaError:
        if audio_codec != "copy":
            destination.unlink(missing_ok=True)
            raise
        destination.unlink(missing_ok=True)
        audio_index = output_command.index("copy", output_command.index("-c:a"))
        output_command[audio_index : audio_index + 1] = ["aac", "-b:a", "192k"]
        try:
            _run(output_command)
        except MediaError:
            destination.unlink(missing_ok=True)
            raise
    return destination


def decode_smoke(path: Path, *, seconds: float, ffmpeg: str = "ffmpeg") -> bool:
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-t",
            f"{seconds:.3f}",
            "-i",
            str(path),
            "-map",
            "0:v:0?",
            "-map",
            "0:a:0?",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def sample_cue_frames(
    path: Path,
    cues: Iterable[Cue],
    destination: Path,
    *,
    count: int,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    cue_list = list(cues)
    if not cue_list or count <= 0:
        return []
    indexes = sorted(
        {
            round(position)
            for position in [
                index * (len(cue_list) - 1) / max(1, count - 1)
                for index in range(count)
            ]
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for output_index, cue_index in enumerate(indexes, start=1):
        cue = cue_list[cue_index]
        timestamp = (cue.start + cue.end) / 2
        frame = destination / f"frame-{output_index:03d}.png"
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(path),
                "-frames:v",
                "1",
                str(frame),
            ]
        )
        frames.append(frame)
    return frames


def lower_frame_difference_ratio(reference: Path, candidate: Path) -> float:
    with (
        Image.open(reference) as reference_image,
        Image.open(candidate) as candidate_image,
    ):
        expected = reference_image.convert("RGB")
        actual = candidate_image.convert("RGB")
        if expected.size != actual.size:
            raise MediaError("sampled frame geometry differs")
        width, height = expected.size
        lower_region = (0, int(height * 0.45), width, height)
        difference = ImageChops.difference(
            expected.crop(lower_region), actual.crop(lower_region)
        )
        histogram = difference.histogram()
        changed_channels = sum(
            sum(histogram[channel * 256 + 25 : (channel + 1) * 256])
            for channel in range(3)
        )
        channel_count = difference.width * difference.height * 3
        return changed_channels / channel_count if channel_count else 0.0


def probe_summary(path: Path, *, ffprobe: str = "ffprobe") -> dict[str, Any]:
    probe = probe_media(path, ffprobe=ffprobe)
    streams = []
    for stream in probe.get("streams", []):
        record = {
            "index": stream.get("index"),
            "codec_type": stream.get("codec_type"),
            "codec_name": stream.get("codec_name"),
        }
        if stream.get("codec_type") == "video":
            record.update(
                {"width": stream.get("width"), "height": stream.get("height")}
            )
        tags = stream.get("tags", {})
        if stream.get("codec_type") == "subtitle" and tags.get("language"):
            record["language"] = tags["language"]
        streams.append(record)
    return {"duration": _duration_from_probe(probe), "streams": streams}

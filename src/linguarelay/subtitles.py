from __future__ import annotations

import re
from collections.abc import Iterable

from .config import SubtitlePolicy
from .models import Cue, TranslationSegment


_SRT_BLOCK = re.compile(
    r"(?ms)^\s*(\d+)\s*\n(\d\d:\d\d:\d\d,\d{3})\s+-->\s+(\d\d:\d\d:\d\d,\d{3})\s*\n(.*?)(?=\n{2,}|\Z)"
)


def _split_long_word(word: str, limit: int) -> list[str]:
    return [word[index : index + limit] for index in range(0, len(word), limit)] or [""]


def _wrap_chunks(text: str, policy: SubtitlePolicy) -> list[str]:
    words: list[str] = []
    for word in text.split():
        words.extend(_split_long_word(word, policy.max_chars_per_line))
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        proposed = word if not current else f"{current} {word}"
        if len(proposed) <= policy.max_chars_per_line:
            current = proposed
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    return [
        "\n".join(lines[index : index + policy.max_lines])
        for index in range(0, len(lines), policy.max_lines)
    ]


def _allocate_durations(
    chunks: list[str], available: float, policy: SubtitlePolicy
) -> list[float]:
    weights = [max(1, len(chunk.replace("\n", ""))) for chunk in chunks]
    total_weight = sum(weights)
    raw = [available * weight / total_weight for weight in weights]
    durations = [
        min(policy.max_cue_seconds, max(policy.min_cue_seconds, value)) for value in raw
    ]
    total = sum(durations)
    if total > available:
        excess = total - available
        for index in sorted(
            range(len(durations)), key=lambda item: durations[item], reverse=True
        ):
            reducible = max(0.0, durations[index] - policy.min_cue_seconds)
            reduction = min(excess, reducible)
            durations[index] -= reduction
            excess -= reduction
            if excess <= 1e-9:
                break
    return durations


def build_cues(
    translated: Iterable[TranslationSegment],
    policy: SubtitlePolicy,
    *,
    media_duration: float,
) -> list[Cue]:
    cues: list[Cue] = []
    previous_end = 0.0
    source_segments = list(translated)
    merged_segments: list[TranslationSegment] = []
    index = 0
    while index < len(source_segments):
        segment = source_segments[index]
        merged = TranslationSegment(
            id=segment.id,
            start=segment.start,
            end=segment.end,
            source_text=segment.source_text,
            text=segment.text,
            uncertainty=list(segment.uncertainty),
            notes=segment.notes,
        )
        while index + 1 < len(source_segments):
            duration = max(0.001, merged.end - merged.start)
            characters = len(merged.text.replace("\n", ""))
            needs_more_time = (
                duration < policy.min_cue_seconds
                or characters / duration > policy.max_chars_per_second
            )
            following = source_segments[index + 1]
            if not needs_more_time or following.start - merged.end > 1.0:
                break
            merged.end = max(merged.end, following.end)
            merged.source_text = f"{merged.source_text} {following.source_text}".strip()
            merged.text = f"{merged.text} {following.text}".strip()
            merged.uncertainty.extend(
                item for item in following.uncertainty if item not in merged.uncertainty
            )
            index += 1
        merged_segments.append(merged)
        index += 1

    for segment in merged_segments:
        if not segment.text.strip():
            continue
        start = max(
            0.0,
            float(segment.start),
            previous_end + (policy.minimum_gap_seconds if cues else 0.0),
        )
        end = min(float(segment.end), media_duration)
        if end <= start:
            continue
        chunks = _wrap_chunks(segment.text, policy)
        if not chunks:
            continue
        gap_total = policy.minimum_gap_seconds * max(0, len(chunks) - 1)
        available = max(0.0, end - start - gap_total)
        if available <= 0:
            continue
        durations = _allocate_durations(chunks, available, policy)
        cursor = start
        for chunk, duration in zip(chunks, durations, strict=True):
            cue_end = min(end, cursor + duration)
            cues.append(Cue(index=len(cues) + 1, start=cursor, end=cue_end, text=chunk))
            previous_end = cue_end
            cursor = cue_end + policy.minimum_gap_seconds
    return cues


def validate_cues(
    cues: Iterable[Cue], policy: SubtitlePolicy, *, media_duration: float
) -> list[str]:
    violations: list[str] = []
    previous_end = 0.0
    for expected_index, cue in enumerate(cues, start=1):
        duration = cue.end - cue.start
        if cue.index != expected_index:
            violations.append(f"cue {expected_index}: non-sequential index")
        if cue.start < 0 or cue.end > media_duration + 0.001 or cue.end <= cue.start:
            violations.append(f"cue {expected_index}: invalid media bounds")
        if expected_index > 1 and cue.start < previous_end - 0.001:
            violations.append(f"cue {expected_index}: overlaps previous cue")
        if (
            expected_index > 1
            and cue.start < previous_end + policy.minimum_gap_seconds - 0.001
        ):
            violations.append(f"cue {expected_index}: gap below policy")
        lines = cue.text.splitlines()
        if not cue.text.strip():
            violations.append(f"cue {expected_index}: empty")
        if len(lines) > policy.max_lines:
            violations.append(f"cue {expected_index}: too many lines")
        if any(len(line) > policy.max_chars_per_line for line in lines):
            violations.append(f"cue {expected_index}: line too long")
        characters = len(cue.text.replace("\n", ""))
        if duration > 0 and characters / duration > policy.max_chars_per_second + 0.01:
            violations.append(f"cue {expected_index}: reading speed")
        if duration > policy.max_cue_seconds + 0.001:
            violations.append(f"cue {expected_index}: duration too long")
        if duration < policy.min_cue_seconds - 0.001:
            violations.append(f"cue {expected_index}: duration too short")
        previous_end = cue.end
    return violations


def text_coverage_matches(
    translated: Iterable[TranslationSegment], cues: Iterable[Cue]
) -> bool:
    expected = " ".join(
        segment.text.strip() for segment in translated if segment.text.strip()
    )
    actual = " ".join(
        cue.text.replace("\n", " ").strip() for cue in cues if cue.text.strip()
    )
    return " ".join(expected.split()) == " ".join(actual.split())


def _format_timestamp(seconds: float, *, decimal: str) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}{decimal}{millis:03d}"


def _parse_timestamp(value: str) -> float:
    hours, minutes, rest = value.replace(",", ".").split(":")
    seconds, milliseconds = rest.split(".")
    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    )


def render_srt(cues: Iterable[Cue]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n{_format_timestamp(cue.start, decimal=',')} --> "
            f"{_format_timestamp(cue.end, decimal=',')}\n{cue.text.strip()}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(cues: Iterable[Cue]) -> str:
    blocks = ["WEBVTT"]
    for cue in cues:
        blocks.append(
            f"{_format_timestamp(cue.start, decimal='.')} --> "
            f"{_format_timestamp(cue.end, decimal='.')}\n{cue.text.strip()}"
        )
    return "\n\n".join(blocks) + "\n"


def parse_srt(value: str) -> list[Cue]:
    normalized = value.replace("\r\n", "\n").strip()
    cues: list[Cue] = []
    for match in _SRT_BLOCK.finditer(normalized):
        cues.append(
            Cue(
                index=int(match.group(1)),
                start=_parse_timestamp(match.group(2)),
                end=_parse_timestamp(match.group(3)),
                text=match.group(4).strip(),
            )
        )
    return cues


def parse_vtt(value: str) -> list[Cue]:
    normalized = value.replace("\r\n", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", normalized):
        lines = [line for line in block.splitlines() if line.strip()]
        timing_index = next(
            (index for index, line in enumerate(lines) if " --> " in line), None
        )
        if timing_index is None:
            continue
        start, end = lines[timing_index].split(" --> ", 1)
        cues.append(
            Cue(
                index=len(cues) + 1,
                start=_parse_timestamp(start.strip()),
                end=_parse_timestamp(end.split()[0].strip()),
                text="\n".join(lines[timing_index + 1 :]).strip(),
            )
        )
    return cues

from linguarelay.config import SubtitlePolicy
from linguarelay.models import TranslationSegment
from linguarelay.subtitles import (
    build_cues,
    parse_srt,
    render_srt,
    render_vtt,
    text_coverage_matches,
    validate_cues,
)


POLICY = SubtitlePolicy(
    max_chars_per_line=24,
    max_lines=2,
    max_chars_per_second=22.0,
    min_cue_seconds=0.8,
    max_cue_seconds=5.0,
    minimum_gap_seconds=0.05,
    font_path="/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    font_size_ratio=0.046,
    bottom_margin_ratio=0.075,
)


def test_build_cues_splits_long_text_without_overlap() -> None:
    translated = [
        TranslationSegment(
            id="s1",
            start=0.0,
            end=8.0,
            source_text="Texto sintético.",
            text="This synthetic sentence is intentionally long enough to require multiple readable cues.",
        )
    ]

    cues = build_cues(translated, POLICY, media_duration=8.0)
    violations = validate_cues(cues, POLICY, media_duration=8.0)

    assert len(cues) >= 2
    assert not violations
    assert all(cues[i].end <= cues[i + 1].start for i in range(len(cues) - 1))
    assert all(
        len(line) <= POLICY.max_chars_per_line
        for cue in cues
        for line in cue.text.splitlines()
    )


def test_srt_vtt_round_trip_and_headers() -> None:
    translated = [
        TranslationSegment(
            id="s1", start=0.2, end=2.5, source_text="Olá.", text="Hello."
        ),
        TranslationSegment(
            id="s2", start=2.7, end=4.8, source_text="Tudo bem?", text="How are you?"
        ),
    ]
    cues = build_cues(translated, POLICY, media_duration=5.0)
    srt = render_srt(cues)
    vtt = render_vtt(cues)

    assert parse_srt(srt) == cues
    assert vtt.startswith("WEBVTT\n\n")
    assert "00:00:00.200 --> 00:00:02.500" in vtt


def test_text_coverage_detects_a_dropped_translation_segment() -> None:
    translated = [
        TranslationSegment(id="s1", start=0.0, end=2.0, source_text="Um.", text="One."),
        TranslationSegment(
            id="s2", start=8.0, end=10.0, source_text="Dois.", text="Two."
        ),
    ]
    cues = build_cues(translated, POLICY, media_duration=5.0)

    assert not text_coverage_matches(translated, cues)

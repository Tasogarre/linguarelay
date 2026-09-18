from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence, TypeVar

from .models import Transcript


T = TypeVar("T")


def _edit_distance(reference: Sequence[T], candidate: Sequence[T]) -> int:
    previous = list(range(len(candidate) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, candidate_item in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != candidate_item),
                )
            )
        previous = current
    return previous[-1]


def _normalized_words(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _normalized_characters(value: str) -> list[str]:
    return list("".join(_normalized_words(value)))


def normalized_wer(reference: str, candidate: str) -> float:
    expected = _normalized_words(reference)
    actual = _normalized_words(candidate)
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def normalized_cer(reference: str, candidate: str) -> float:
    expected = _normalized_characters(reference)
    actual = _normalized_characters(candidate)
    if not expected:
        return 0.0 if not actual else 1.0
    return _edit_distance(expected, actual) / len(expected)


def _recall(expected: Sequence[str], candidate: str) -> float:
    if not expected:
        return 1.0
    candidate_words = _normalized_words(candidate)

    def contains_term(term: str) -> bool:
        term_words = _normalized_words(term)
        if not term_words:
            return False
        width = len(term_words)
        return any(
            candidate_words[index : index + width] == term_words
            for index in range(0, len(candidate_words) - width + 1)
        )

    found = sum(contains_term(item) for item in expected)
    return found / len(expected)


def _punctuation_f1(reference: str, candidate: str) -> float:
    expected = Counter(
        character
        for character in reference
        if unicodedata.category(character).startswith("P")
    )
    actual = Counter(
        character
        for character in candidate
        if unicodedata.category(character).startswith("P")
    )
    if not expected:
        return 1.0 if not actual else 0.0
    overlap = sum((expected & actual).values())
    precision = overlap / sum(actual.values()) if actual else 0.0
    recall = overlap / sum(expected.values())
    return (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )


def _speech_intervals(transcript: Transcript) -> list[tuple[float, float]]:
    intervals = sorted(
        (segment.start, segment.end)
        for segment in transcript.segments
        if math.isfinite(segment.start)
        and math.isfinite(segment.end)
        and segment.end > segment.start
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _timing_iou(reference: Transcript, candidate: Transcript) -> float:
    expected = _speech_intervals(reference)
    actual = _speech_intervals(candidate)
    if not expected or not actual:
        return 1.0 if expected == actual else 0.0
    intersection = 0.0
    left = right = 0
    while left < len(expected) and right < len(actual):
        intersection += max(
            0.0,
            min(expected[left][1], actual[right][1])
            - max(expected[left][0], actual[right][0]),
        )
        if expected[left][1] <= actual[right][1]:
            left += 1
        else:
            right += 1
    expected_duration = sum(end - start for start, end in expected)
    actual_duration = sum(end - start for start, end in actual)
    union = expected_duration + actual_duration - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True, slots=True)
class TranscriptEvaluation:
    wer: float
    cer: float
    glossary_term_recall: float
    number_recall: float
    punctuation_f1: float
    timing_iou: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def evaluate_transcript(
    reference: Transcript,
    candidate: Transcript,
    *,
    glossary_terms: Sequence[str] = (),
    expected_numbers: Sequence[str] = (),
) -> TranscriptEvaluation:
    return TranscriptEvaluation(
        wer=normalized_wer(reference.text, candidate.text),
        cer=normalized_cer(reference.text, candidate.text),
        glossary_term_recall=_recall(glossary_terms, candidate.text),
        number_recall=_recall(expected_numbers, candidate.text),
        punctuation_f1=_punctuation_f1(reference.text, candidate.text),
        timing_iou=_timing_iou(reference, candidate),
    )

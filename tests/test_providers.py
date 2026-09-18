import json
from pathlib import Path

import pytest

from linguarelay.metrics import evaluate_transcript
from linguarelay.models import Transcript
from linguarelay.providers import parse_deepgram_response, parse_openai_whisper_response
from linguarelay.providers import NativeCaptionProvider


def test_parse_deepgram_preserves_word_confidence_and_uncertainty() -> None:
    payload = {
        "results": {
            "utterances": [
                {
                    "start": 0.1,
                    "end": 1.7,
                    "confidence": 0.72,
                    "transcript": "Olá, Ana.",
                    "words": [
                        {
                            "word": "Olá",
                            "punctuated_word": "Olá,",
                            "start": 0.1,
                            "end": 0.5,
                            "confidence": 0.91,
                        },
                        {
                            "word": "Ana",
                            "punctuated_word": "Ana.",
                            "start": 0.6,
                            "end": 1.7,
                            "confidence": 0.42,
                        },
                    ],
                }
            ]
        }
    }

    transcript = parse_deepgram_response(
        payload, provider="deepgram_nova3", model="nova-3", min_confidence=0.65
    )

    assert transcript.text == "Olá, Ana."
    assert transcript.segments[0].words[1].confidence == 0.42
    assert "low_word_confidence" in transcript.segments[0].uncertainty


def test_parse_openai_whisper_flags_failed_logprob() -> None:
    payload = {
        "language": "portuguese",
        "duration": 2.0,
        "text": "Olá, Ana.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": " Olá, Ana.",
                "avg_logprob": -1.2,
                "compression_ratio": 1.1,
                "no_speech_prob": 0.01,
            }
        ],
    }

    transcript = parse_openai_whisper_response(
        payload, provider="openai_whisper1", model="whisper-1"
    )

    assert transcript.text == "Olá, Ana."
    assert "low_log_probability" in transcript.segments[0].uncertainty


def test_asr_parsers_reject_non_finite_provider_numbers() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        parse_openai_whisper_response(
            {
                "segments": [
                    {
                        "start": 0,
                        "end": 1,
                        "text": "teste",
                        "avg_logprob": float("nan"),
                    }
                ]
            },
            provider="openai",
            model="whisper-1",
        )

    with pytest.raises(ValueError, match="non-finite"):
        parse_deepgram_response(
            {
                "results": {
                    "utterances": [
                        {
                            "start": 0,
                            "end": 1,
                            "transcript": "teste",
                            "confidence": float("inf"),
                        }
                    ]
                }
            },
            provider="deepgram",
            model="nova-3",
            min_confidence=0.65,
        )


def test_native_caption_fixture_matches_synthetic_reference() -> None:
    provider = NativeCaptionProvider(
        name="native",
        model="synthetic-track",
        caption_path=Path("tests/fixtures/native.vtt"),
    )
    candidate = provider.transcribe(
        Path("unused.wav"),
        language="pt-BR",
        min_confidence=0.65,
    )
    reference = Transcript.from_dict(
        json.loads(Path("tests/fixtures/reference.json").read_text(encoding="utf-8"))
    )

    metrics = evaluate_transcript(
        reference,
        candidate,
        glossary_terms=["Ana", "LinguaRelay"],
        expected_numbers=["42"],
    )

    assert metrics.wer == 0
    assert metrics.cer == 0
    assert metrics.glossary_term_recall == 1
    assert metrics.number_recall == 1
    assert metrics.timing_iou == 1

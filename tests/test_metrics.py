from linguarelay.metrics import evaluate_transcript, normalized_cer, normalized_wer
from linguarelay.models import Segment, Transcript


def _transcript(texts: list[str], starts: list[float] | None = None) -> Transcript:
    starts = starts or [float(i * 2) for i in range(len(texts))]
    return Transcript(
        language="pt-BR",
        provider="fixture",
        model="fixture",
        segments=[
            Segment(id=f"s-{i}", start=start, end=start + 1.8, text=text)
            for i, (start, text) in enumerate(zip(starts, texts, strict=True))
        ],
    )


def test_wer_and_cer_are_zero_for_equivalent_case_and_spacing() -> None:
    assert normalized_wer(" Olá,  MUNDO! ", "olá mundo") == 0.0
    assert normalized_cer(" Olá,  MUNDO! ", "olá mundo") == 0.0


def test_transcript_evaluation_catches_term_number_and_timing_errors() -> None:
    reference = _transcript(
        ["Olá, meu nome é Ana.", "O LinguaRelay processa 42 exemplos."], [0.0, 2.1]
    )
    candidate = _transcript(
        ["Olá, meu nome é Jana.", "O sistema processa exemplos."], [0.7, 3.0]
    )

    result = evaluate_transcript(
        reference,
        candidate,
        glossary_terms=["Ana", "LinguaRelay"],
        expected_numbers=["42"],
    )

    assert result.wer > 0
    assert result.cer > 0
    assert result.glossary_term_recall == 0.0
    assert result.number_recall == 0.0
    assert result.timing_iou < 1.0


def test_timing_iou_is_independent_of_provider_segmentation() -> None:
    reference = _transcript(["um", "dois"], [0.0, 2.0])
    candidate = Transcript(
        language="pt-BR",
        provider="fixture",
        model="fixture",
        segments=[
            Segment(id="a", start=0.0, end=0.9, text="um"),
            Segment(id="b", start=0.9, end=1.8, text=""),
            Segment(id="c", start=2.0, end=3.8, text="dois"),
        ],
    )

    result = evaluate_transcript(reference, candidate)

    assert result.timing_iou == 1.0

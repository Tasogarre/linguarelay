from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Word":
        return cls(
            text=str(value["text"]),
            start=float(value["start"]),
            end=float(value["end"]),
            confidence=float(value["confidence"])
            if value.get("confidence") is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    id: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    uncertainty: list[str] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(
            id=str(value["id"]),
            start=float(value["start"]),
            end=float(value["end"]),
            text=str(value["text"]).strip(),
            confidence=float(value["confidence"])
            if value.get("confidence") is not None
            else None,
            uncertainty=[str(item) for item in value.get("uncertainty", [])],
            words=[Word.from_dict(item) for item in value.get("words", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Transcript:
    language: str
    provider: str
    model: str
    segments: list[Segment]
    source: str = "asr"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return " ".join(
            segment.text.strip() for segment in self.segments if segment.text.strip()
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Transcript":
        return cls(
            language=str(value["language"]),
            provider=str(value.get("provider", "unknown")),
            model=str(value.get("model", "unknown")),
            source=str(value.get("source", "asr")),
            metadata=dict(value.get("metadata", {})),
            segments=[Segment.from_dict(item) for item in value.get("segments", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "language": self.language,
            "provider": self.provider,
            "model": self.model,
            "source": self.source,
            "metadata": self.metadata,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(slots=True)
class TranslationSegment:
    id: str
    start: float
    end: float
    source_text: str
    text: str
    uncertainty: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TranslationSegment":
        return cls(
            id=str(value["id"]),
            start=float(value["start"]),
            end=float(value["end"]),
            source_text=str(value.get("source_text", "")),
            text=str(value["text"]).strip(),
            uncertainty=[str(item) for item in value.get("uncertainty", [])],
            notes=str(value.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Cue:
    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

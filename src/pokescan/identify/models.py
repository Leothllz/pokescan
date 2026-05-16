"""Shared data models for card identification."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OCRResult:
    """Text extracted from targeted zones of a card image."""

    name: str | None = None
    collector_number: str | None = None  # "169/203"
    hp: str | None = None                # "210"
    language: str | None = None          # "fr", "en"
    year: str | None = None              # "2021"
    raw_texts: dict[str, str] = field(default_factory=dict)

    @property
    def local_id(self) -> str | None:
        """Return the card number part (before the slash)."""
        if self.collector_number and "/" in self.collector_number:
            return self.collector_number.split("/")[0].strip()
        return self.collector_number

    @property
    def set_total(self) -> str | None:
        """Return the set total part (after the slash)."""
        if self.collector_number and "/" in self.collector_number:
            return self.collector_number.split("/")[1].strip()
        return None


@dataclass
class CardCandidate:
    """A potential match from the TCGdex database."""

    card_id: str              # "swsh7-169"
    name: str                 # "Pyroli V"
    set_name: str             # "Évolution Céleste"
    set_id: str               # "swsh7"
    number: str               # "169"
    number_total: str | None  # "203"
    rarity: str | None = None
    language: str = "fr"
    image_url: str | None = None
    hp: int | None = None
    pricing: dict | None = None
    score: float = 0.0
    score_detail: dict = field(default_factory=dict)


@dataclass
class CardIdentity:
    """Final identification result combining OCR + database matching."""

    best_match: CardCandidate | None
    confidence: float
    ocr_result: OCRResult
    candidates: list[CardCandidate] = field(default_factory=list)

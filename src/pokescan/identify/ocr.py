"""OCR extraction from targeted card zones using EasyOCR.

Zones:
    - TOP (0-12%): card name
    - TOP-RIGHT (0-8%, right 40%): HP value
    - BOTTOM (85-100%): collector number, copyright, year
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from pokescan.identify.models import OCRResult

# Lazy-loaded EasyOCR reader (heavy import).
_reader = None


def _get_reader(languages: list[str] | None = None):
    """Return a cached EasyOCR reader instance."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(
            languages or ["fr", "en"],
            gpu=False,
            verbose=False,
        )
    return _reader


# ---------------------------------------------------------------------------
# Zone definitions (relative to card image height/width)
# ---------------------------------------------------------------------------

ZONE_NAME = (0.0, 0.0, 1.0, 0.13)         # top strip: full width, 13% height
ZONE_HP = (0.55, 0.0, 1.0, 0.10)           # top-right corner
ZONE_BOTTOM = (0.0, 0.84, 1.0, 1.0)        # bottom strip: 16% height


def _crop_zone(image: np.ndarray, zone: tuple[float, float, float, float]) -> np.ndarray:
    """Crop a zone from the image using relative coordinates (x1, y1, x2, y2)."""
    h, w = image.shape[:2]
    x1 = int(w * zone[0])
    y1 = int(h * zone[1])
    x2 = int(w * zone[2])
    y2 = int(h * zone[3])
    return image[y1:y2, x1:x2]


def _preprocess_zone(crop: np.ndarray) -> np.ndarray:
    """Preprocess a zone crop for better OCR accuracy."""
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # Upscale small crops for better OCR.
    h, w = gray.shape[:2]
    if w < 300:
        scale = 300 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Denoise + adaptive threshold.
    gray = cv2.fastNlMeansDenoising(gray, h=12)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4,
    )
    return gray


def _read_zone(image: np.ndarray, zone: tuple[float, float, float, float]) -> str:
    """Crop, preprocess, and OCR a zone. Return raw text."""
    crop = _crop_zone(image, zone)
    if crop.size == 0:
        return ""
    processed = _preprocess_zone(crop)
    reader = _get_reader()
    results = reader.readtext(processed, detail=0, paragraph=True)
    return " ".join(results).strip()


# ---------------------------------------------------------------------------
# Regex patterns for post-processing
# ---------------------------------------------------------------------------

RE_COLLECTOR = re.compile(r"(\d{1,4})\s*/\s*(\d{1,4})")
RE_HP = re.compile(r"(\d{2,3})\s*(?:HP|PV|hp|pv)", re.IGNORECASE)
RE_HP_ALT = re.compile(r"(?:HP|PV|hp|pv)\s*(\d{2,3})", re.IGNORECASE)
RE_YEAR = re.compile(r"[©®]?\s*((?:19|20)\d{2})")
RE_POKEMON_COMPANY = re.compile(
    r"(?:Pok[eé]mon|Nintendo|Creatures|GAME\s*FREAK)", re.IGNORECASE,
)


def _detect_language(bottom_text: str, name_text: str) -> str | None:
    """Heuristic language detection from card text."""
    combined = f"{bottom_text} {name_text}".lower()
    if " pv " in combined or "pv" in combined.split():
        return "fr"
    if " hp " in combined or "hp" in combined.split():
        return "en"
    # French-specific patterns.
    fr_indicators = ["faiblesse", "résistance", "retraite", "énergie", "attaque"]
    if any(ind in combined for ind in fr_indicators):
        return "fr"
    en_indicators = ["weakness", "resistance", "retreat", "energy", "attack"]
    if any(ind in combined for ind in en_indicators):
        return "en"
    return None


def _clean_card_name(raw: str) -> str | None:
    """Clean OCR noise from the card name."""
    if not raw:
        return None
    # Remove common OCR artifacts.
    name = re.sub(r"[|_\[\]{}]", "", raw)
    # Remove lone numbers at start/end.
    name = re.sub(r"^\d+\s+", "", name)
    name = re.sub(r"\s+\d+$", "", name)
    name = name.strip()
    # Must have at least 2 chars to be a valid name.
    return name if len(name) >= 2 else None


def extract_card_text(image: np.ndarray) -> OCRResult:
    """Run OCR on targeted zones and extract structured card information.

    Args:
        image: BGR card image (already cropped from slab if applicable).

    Returns:
        OCRResult with extracted fields.
    """
    raw_texts: dict[str, str] = {}

    # 1. Name zone.
    name_raw = _read_zone(image, ZONE_NAME)
    raw_texts["name"] = name_raw

    # 2. HP zone.
    hp_raw = _read_zone(image, ZONE_HP)
    raw_texts["hp"] = hp_raw

    # 3. Bottom zone (collector number, year, copyright).
    bottom_raw = _read_zone(image, ZONE_BOTTOM)
    raw_texts["bottom"] = bottom_raw

    # --- Post-processing ---

    # Card name.
    name = _clean_card_name(name_raw)

    # Collector number.
    collector_number = None
    match = RE_COLLECTOR.search(bottom_raw)
    if match:
        collector_number = f"{match.group(1)}/{match.group(2)}"

    # HP.
    hp = None
    for pattern in (RE_HP, RE_HP_ALT):
        m = pattern.search(hp_raw)
        if not m:
            m = pattern.search(name_raw)
        if m:
            hp = m.group(1)
            break

    # Year.
    year = None
    m = RE_YEAR.search(bottom_raw)
    if m:
        year = m.group(1)

    # Language.
    language = _detect_language(bottom_raw, name_raw)

    return OCRResult(
        name=name,
        collector_number=collector_number,
        hp=hp,
        language=language,
        year=year,
        raw_texts=raw_texts,
    )

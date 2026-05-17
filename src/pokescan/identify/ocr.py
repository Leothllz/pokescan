"""OCR extraction from targeted card zones using EasyOCR.

Zones:
    - TOP (0-12%): card name
    - TOP-RIGHT (0-8%, right 40%): HP value
    - BOTTOM (85-100%): collector number, copyright, year
"""

from __future__ import annotations

import re
import os
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np

from pokescan.identify.models import OCRResult

# Lazy-loaded EasyOCR reader (heavy import).
_reader = None


def _is_windows_rocm_runtime() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return "+rocm" in metadata.version("torch").lower()
    except metadata.PackageNotFoundError:
        return False


def _sidecar_python_command() -> list[str]:
    configured = os.environ.get("POKESCAN_CPU_PYTHON")
    if configured:
        return [configured]
    if sys.platform == "win32":
        return ["py", "-3.11"]
    return [sys.executable]


def _extract_card_text_sidecar(image: np.ndarray) -> OCRResult | None:
    if os.environ.get("POKESCAN_OCR_SIDECAR_ACTIVE") == "1":
        return None

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp, image)
        image_path = tmp.name

    script = r"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("src"))
from pokescan.identify.ocr import extract_card_text

image = np.load(sys.argv[1])
result = extract_card_text(image)
print(json.dumps({
    "name": result.name,
    "collector_number": result.collector_number,
    "hp": result.hp,
    "language": result.language,
    "year": result.year,
    "raw_texts": result.raw_texts,
}, ensure_ascii=False))
"""
    env = os.environ.copy()
    env["POKESCAN_OCR_SIDECAR_ACTIVE"] = "1"
    try:
        completed = subprocess.run(
            [*_sidecar_python_command(), "-c", script, image_path],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        data = json.loads(completed.stdout.strip().splitlines()[-1])
        return OCRResult(**data)
    except Exception:
        return None
    finally:
        try:
            Path(image_path).unlink()
        except OSError:
            pass


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

ZONE_NAME = (0.0, 0.0, 1.0, 0.16)         # top strip: name, level, HP
ZONE_HP = (0.55, 0.0, 1.0, 0.16)           # top-right corner
ZONE_BODY_TEXT = (0.0, 0.35, 1.0, 0.78)    # attack/rules text for language hints
ZONE_BOTTOM = (0.0, 0.88, 1.0, 1.0)        # footer strip: collector/year
ZONE_FOOTER = (0.0, 0.82, 1.0, 1.0)        # wider footer fallback for tiny numbers
ZONE_NUMBER = (0.72, 0.92, 1.0, 1.0)       # bottom-right collector number
ZONE_NUMBER_ALT = (0.66, 0.62, 1.0, 0.84)  # card footer when fallback crop includes table


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


def _read_zone_lines(image: np.ndarray, zone: tuple[float, float, float, float]) -> list[str]:
    """Crop, preprocess, and OCR a zone. Return text lines in reading order."""
    crop = _crop_zone(image, zone)
    if crop.size == 0:
        return []
    processed = _preprocess_zone(crop)
    reader = _get_reader()
    results = reader.readtext(processed, detail=1, paragraph=False)
    lines: list[tuple[float, float, str]] = []
    for result in results:
        if len(result) < 2:
            continue
        box, text = result[0], str(result[1]).strip()
        if not text:
            continue
        try:
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            lines.append((min(ys), min(xs), text))
        except (TypeError, IndexError):
            lines.append((0.0, 0.0, text))
    return [text for _y, _x, text in sorted(lines)]


def _read_zone(image: np.ndarray, zone: tuple[float, float, float, float]) -> str:
    """Crop, preprocess, and OCR a zone. Return raw text."""
    return " ".join(_read_zone_lines(image, zone)).strip()


def _read_zone_raw(image: np.ndarray, zone: tuple[float, float, float, float]) -> str:
    """OCR a zone without thresholding, useful for tiny collector numbers."""
    crop = _crop_zone(image, zone)
    if crop.size == 0:
        return ""
    reader = _get_reader()
    results = reader.readtext(crop, detail=0, paragraph=False)
    return " ".join(str(text).strip() for text in results if str(text).strip())


# ---------------------------------------------------------------------------
# Regex patterns for post-processing
# ---------------------------------------------------------------------------

RE_COLLECTOR = re.compile(r"([0-9OIlS]{1,4})\s*/\s*([0-9OIlS]{1,4})")
RE_COLLECTOR_HASH = re.compile(r"(?:#|No\.?|N[°o])\s*([0-9OIlS]{1,4})\b", re.IGNORECASE)
RE_HP = re.compile(r"(\d{2,3})\s*(?:HP|PV|P)\b", re.IGNORECASE)
RE_HP_ALT = re.compile(r"\b(?:HP|PV|P)\s*(\d{2,3})", re.IGNORECASE)
RE_YEAR = re.compile(r"(?:19|20)\d{2}")
RE_POKEMON_COMPANY = re.compile(
    r"(?:Pok[eé]mon|Nintendo|Creatures|GAME\s*FREAK)", re.IGNORECASE,
)


def _detect_language(bottom_text: str, name_text: str, body_text: str = "") -> str | None:
    """Heuristic language detection from card text."""
    combined = f"{bottom_text} {name_text} {body_text}".lower()
    if re.search(r"(?:\bpv\b|\d{2,3}\s*pv\b|\bpv\s*\d{2,3})", combined):
        return "fr"
    if re.search(r"(?:\bhp\b|\d{2,3}\s*hp\b|\bhp\s*\d{2,3})", combined):
        return "en"
    # French-specific patterns.
    fr_indicators = ["faiblesse", "résistance", "retraite", "énergie", "attaque"]
    if any(ind in combined for ind in fr_indicators):
        return "fr"
    en_indicators = ["weakness", "resistance", "retreat", "energy", "attack"]
    if any(ind in combined for ind in en_indicators):
        return "en"
    return None


def _normalize_ocr_number(raw: str) -> str:
    """Normalize common OCR confusions in numeric card ids."""
    return raw.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "s": "5"}))


def _clean_card_name(raw: str) -> str | None:
    """Clean OCR noise from the card name."""
    if not raw:
        return None
    # Remove common OCR artifacts.
    name = re.sub(r"[|_\[\]{}]", "", raw)
    name = re.sub(r"\b(?:niv|n[ilv]?|miv|miy|nivcau|niveau)\.?\s*\d+\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+(?:[a-z]{1,3}\.?\s*){1,3}\d{1,3}\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b\d{2,3}\s*(?:HP|PV|P)\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\b(?:HP|PV|P)\s*\d{2,3}\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bBASE\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*[%#].*$", "", name)
    name = re.sub(r"\b(?:Basic|Baslc)\s+Pok[eé]mon\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\bPok[eé]mon\s+de\s+base\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"^\s*(?:Stage|Niveau)\s*\d+\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r",.*$", "", name)
    # Remove lone numbers at start/end.
    name = re.sub(r"^\d+\s+", "", name)
    name = re.sub(r"\s+\d+$", "", name)
    name = name.strip()
    # Must have at least 2 chars to be a valid name.
    return name if len(name) >= 2 else None


def _extract_year(raw: str) -> str | None:
    """Extract a plausible card copyright/release year."""
    current_max = datetime.now().year + 1
    years = [year for year in RE_YEAR.findall(raw) if 1996 <= int(year) <= current_max]
    return years[-1] if years else None


def _extract_hp_from_text(*texts: str) -> str | None:
    """Extract HP/PV from the first OCR text that contains it."""
    for raw in texts:
        for pattern in (RE_HP, RE_HP_ALT):
            m = pattern.search(raw or "")
            if m:
                return m.group(1)
    return None


def extract_card_text(image: np.ndarray) -> OCRResult:
    """Run OCR on targeted zones and extract structured card information.

    Args:
        image: BGR card image (already cropped from slab if applicable).

    Returns:
        OCRResult with extracted fields.
    """
    if _is_windows_rocm_runtime():
        sidecar_result = _extract_card_text_sidecar(image)
        if sidecar_result is not None:
            return sidecar_result

    raw_texts: dict[str, str] = {}

    # 1. Read the top strip first; raw OCR is often enough and much faster.
    name_parts = [_read_zone_raw(image, ZONE_NAME)]
    hp_parts = [_read_zone_raw(image, ZONE_HP)]
    name_raw = " ".join(text for text in name_parts if text).strip()
    hp_raw = " ".join(text for text in hp_parts if text).strip()
    name = _clean_card_name(name_raw)
    hp = _extract_hp_from_text(hp_raw, name_raw)

    # 2. Fall back to thresholded top OCR only when the fast pass is incomplete.
    if not name or not hp:
        name_processed = _read_zone(image, ZONE_NAME)
        if name_processed:
            name_parts.append(name_processed)
            name_raw = " ".join(text for text in name_parts if text).strip()
            name = name or _clean_card_name(name_raw)
            hp = hp or _extract_hp_from_text(hp_raw, name_raw)

    if not hp:
        hp_processed = _read_zone(image, ZONE_HP)
        if hp_processed:
            hp_parts.append(hp_processed)
            hp_raw = " ".join(text for text in hp_parts if text).strip()
            hp = _extract_hp_from_text(hp_raw, name_raw)

    raw_texts["name"] = name_raw
    raw_texts["hp"] = hp_raw

    # 3. Body OCR is expensive; only read it if the top strip did not reveal language.
    language = _detect_language("", name_raw, hp_raw)
    body_raw = ""
    if language is None:
        body_raw = _read_zone_raw(image, ZONE_BODY_TEXT)
        language = _detect_language("", name_raw, body_raw)
    raw_texts["body"] = body_raw

    # 4. Bottom zones (collector number, year, copyright).
    bottom_raw = _read_zone(image, ZONE_BOTTOM)
    raw_texts["bottom"] = bottom_raw
    number_raw = " ".join(
        text for text in [
            _read_zone(image, ZONE_NUMBER),
            _read_zone_raw(image, ZONE_NUMBER),
            _read_zone_raw(image, ZONE_NUMBER_ALT),
        ] if text
    ).strip()
    raw_texts["number"] = number_raw

    # --- Post-processing ---

    # Collector number.
    collector_number = None
    footer_raw = ""
    match = RE_COLLECTOR.search(number_raw) or RE_COLLECTOR.search(bottom_raw)
    if not match:
        footer_raw = _read_zone_raw(image, ZONE_FOOTER)
        raw_texts["footer"] = footer_raw
        match = RE_COLLECTOR.search(footer_raw)
    if match:
        collector_number = f"{_normalize_ocr_number(match.group(1))}/{_normalize_ocr_number(match.group(2))}"
    else:
        match = RE_COLLECTOR_HASH.search(number_raw) or RE_COLLECTOR_HASH.search(footer_raw)
        if match:
            collector_number = _normalize_ocr_number(match.group(1))

    # Year.
    year = _extract_year(f"{bottom_raw} {footer_raw}")

    # Language.
    language = language or _detect_language(f"{bottom_raw} {footer_raw}", name_raw, body_raw)

    return OCRResult(
        name=name,
        collector_number=collector_number,
        hp=hp,
        language=language,
        year=year,
        raw_texts=raw_texts,
    )

"""TCGdex API client for card database search.

API documentation: https://tcgdex.dev/rest
Base URL: https://api.tcgdex.net/v2/{lang}/...
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from pokescan.identify.models import CardCandidate, OCRResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TCGDEX_BASE = "https://api.tcgdex.net/v2"
CACHE_TTL_SECONDS = 86400  # 24 hours

_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "Accept": "application/json",
            "User-Agent": "PokeScan/1.0",
        })
    return _session


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class _Cache:
    """Simple file-backed JSON cache for TCGdex responses."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        if time.time() - entry.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return entry.get("data")

    def put(self, key: str, data: Any) -> None:
        self._data[key] = {"ts": time.time(), "data": data}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
        except OSError:
            pass


_cache: _Cache | None = None


def _get_cache() -> _Cache:
    global _cache
    if _cache is None:
        from pokescan.paths import DATA_DIR
        _cache = _Cache(DATA_DIR / "tcgdex_cache.json")
    return _cache


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(path: str, params: dict | None = None) -> Any | None:
    """GET from TCGdex API with caching."""
    cache = _get_cache()
    cache_key = f"{path}|{json.dumps(params or {}, sort_keys=True)}"

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{TCGDEX_BASE}{path}"
    try:
        resp = _get_session().get(url, params=params, timeout=15)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        cache.put(cache_key, data)
        return data
    except (requests.RequestException, json.JSONDecodeError):
        return None


def _brief_to_candidate(brief: dict, language: str) -> CardCandidate:
    """Convert a TCGdex CardBrief to a CardCandidate (minimal data)."""
    return CardCandidate(
        card_id=brief.get("id", ""),
        name=brief.get("name", ""),
        set_name="",
        set_id=brief.get("id", "").rsplit("-", 1)[0] if "-" in brief.get("id", "") else "",
        number=brief.get("localId", ""),
        number_total=None,
        language=language,
        image_url=brief.get("image"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_by_name(name: str, language: str = "fr") -> list[CardCandidate]:
    """Search TCGdex for cards matching a name."""
    data = _api_get(f"/{language}/cards", params={"name": name})
    if not data or not isinstance(data, list):
        return []
    return [_brief_to_candidate(item, language) for item in data]


def get_card_detail(card_id: str, language: str = "fr") -> dict | None:
    """Fetch full card detail including pricing."""
    return _api_get(f"/{language}/cards/{card_id}")


def enrich_candidate(candidate: CardCandidate) -> CardCandidate:
    """Fetch full details for a candidate and fill in missing fields."""
    detail = get_card_detail(candidate.card_id, candidate.language)
    if not detail:
        return candidate

    set_info = detail.get("set", {})
    card_count = set_info.get("cardCount", {})

    candidate.set_name = set_info.get("name", candidate.set_name)
    candidate.set_id = set_info.get("id", candidate.set_id)
    candidate.number_total = str(card_count.get("official", "")) or None
    candidate.rarity = detail.get("rarity", candidate.rarity)
    candidate.hp = detail.get("hp")
    candidate.image_url = detail.get("image", candidate.image_url)
    candidate.pricing = detail.get("pricing")

    return candidate


def search_tcgdex(ocr: OCRResult, language: str = "fr") -> list[CardCandidate]:
    """Search TCGdex using OCR results with multi-strategy fallback.

    Strategy:
        1. Search by name in the detected language.
        2. If language is FR and no results, retry in EN.
        3. Filter by localId if collector number available.
    """
    lang = ocr.language or language
    candidates: list[CardCandidate] = []

    if ocr.name:
        # Strategy 1: search in detected language.
        candidates = search_by_name(ocr.name, lang)

        # Strategy 2: fallback to other language.
        if not candidates and lang == "fr":
            candidates = search_by_name(ocr.name, "en")
        elif not candidates and lang == "en":
            candidates = search_by_name(ocr.name, "fr")

        # Strategy 2b: try without suffix (V, VMAX, EX, etc.)
        if not candidates:
            # Strip common suffixes.
            import re
            base_name = re.sub(
                r"\s+(?:V|VMAX|VSTAR|EX|GX|ex|GX|Tag\s*Team|-EX|-GX)\s*$",
                "",
                ocr.name,
                flags=re.IGNORECASE,
            ).strip()
            if base_name != ocr.name:
                candidates = search_by_name(base_name, lang)

    # Filter by collector number if available.
    if candidates and ocr.local_id:
        exact_matches = [
            c for c in candidates
            if c.number == ocr.local_id
        ]
        if exact_matches:
            # Put exact number matches first, keep others as fallback.
            remaining = [c for c in candidates if c not in exact_matches]
            candidates = exact_matches + remaining

    return candidates


def enrich_candidates(
    candidates: list[CardCandidate],
    max_enrich: int = 10,
) -> list[CardCandidate]:
    """Enrich the top candidates with full details (HP, pricing, set info)."""
    enriched = []
    for i, candidate in enumerate(candidates):
        if i < max_enrich:
            enriched.append(enrich_candidate(candidate))
        else:
            enriched.append(candidate)
    return enriched

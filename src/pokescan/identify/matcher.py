"""Score fusion for card candidate matching.

Computes a weighted score combining OCR similarity and database fields.
"""

from __future__ import annotations

from pokescan.identify.models import CardCandidate, OCRResult

# Weights Phase 1 (OCR + DB only — used when no visual index).
WEIGHTS_OCR_ONLY = {
    "name": 0.40,
    "collector_number": 0.35,
    "hp": 0.10,
    "year": 0.10,
    "set_total": 0.05,
}

# Weights Phase 2 (OCR + visual embeddings).
WEIGHTS_WITH_VISUAL = {
    "name": 0.30,
    "collector_number": 0.30,
    "visual": 0.25,
    "hp": 0.05,
    "year": 0.05,
    "set_total": 0.05,
}


def _fuzzy_ratio(a: str | None, b: str | None) -> float:
    """Return 0.0–1.0 fuzzy similarity between two strings."""
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio(a.lower().strip(), b.lower().strip()) / 100.0
    except ImportError:
        # Fallback to basic comparison.
        a_clean = a.lower().strip()
        b_clean = b.lower().strip()
        if a_clean == b_clean:
            return 1.0
        if a_clean in b_clean or b_clean in a_clean:
            return 0.7
        return 0.0


def score_candidate(
    ocr: OCRResult,
    candidate: CardCandidate,
    visual_score: float | None = None,
) -> float:
    """Compute a matching score for a candidate against OCR results.

    Args:
        ocr: OCR extraction results.
        candidate: Card candidate from TCGdex.
        visual_score: Optional CLIP similarity score in [0, 1].

    Returns a float in [0.0, 1.0].
    """
    scores: dict[str, float] = {}

    # 1. Name similarity.
    scores["name"] = _fuzzy_ratio(ocr.name, candidate.name)

    # 2. Collector number match.
    if ocr.local_id and candidate.number:
        # Normalize: strip leading zeros for comparison.
        ocr_num = ocr.local_id.lstrip("0") or "0"
        cand_num = candidate.number.lstrip("0") or "0"
        scores["collector_number"] = 1.0 if ocr_num == cand_num else 0.0
    else:
        scores["collector_number"] = 0.0

    # 3. HP match.
    if ocr.hp and candidate.hp is not None:
        scores["hp"] = 1.0 if str(candidate.hp) == ocr.hp else 0.0
    else:
        # No penalty if HP is unavailable.
        scores["hp"] = 0.5

    # 4. Year (approximate via set release, not directly in TCGdex brief).
    # For now, give neutral score.
    scores["year"] = 0.5

    # 5. Set total match.
    if ocr.set_total and candidate.number_total:
        scores["set_total"] = 1.0 if ocr.set_total == candidate.number_total else 0.0
    else:
        scores["set_total"] = 0.5

    # 6. Visual similarity (Phase 2).
    if visual_score is not None:
        scores["visual"] = visual_score
        weights = WEIGHTS_WITH_VISUAL
    else:
        weights = WEIGHTS_OCR_ONLY

    # Weighted sum.
    total = sum(weights.get(k, 0) * scores.get(k, 0.5) for k in weights)

    # Store detail on the candidate.
    candidate.score_detail = scores
    candidate.score = total

    return total


def score_candidates(
    ocr: OCRResult,
    candidates: list[CardCandidate],
    visual_scores: dict[str, float] | None = None,
) -> list[CardCandidate]:
    """Score and sort candidates by descending match quality.

    Args:
        ocr: OCR results.
        candidates: List of candidates to score.
        visual_scores: Optional dict mapping card_id → CLIP similarity score.
    """
    for candidate in candidates:
        vis = visual_scores.get(candidate.card_id) if visual_scores else None
        score_candidate(ocr, candidate, visual_score=vis)
    return sorted(candidates, key=lambda c: c.score, reverse=True)

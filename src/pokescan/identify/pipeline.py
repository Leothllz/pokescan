"""Card identification pipeline — orchestrates crop → OCR → search → score.

Usage:
    from pokescan.identify import identify_card
    result = identify_card(Path("photo.jpg"))
    print(result.best_match.name, result.confidence)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from pokescan.identify.models import CardIdentity, OCRResult


def identify_card(
    image_or_path: Path | str | np.ndarray,
    *,
    language: str = "fr",
    top_k: int = 5,
    skip_crop: bool = False,
) -> CardIdentity:
    """Run the full identification pipeline on a card image.

    Args:
        image_or_path: Path to an image file, or a BGR numpy array.
        language: Primary language for TCGdex search ("fr" or "en").
        top_k: Number of top candidates to return.
        skip_crop: If True, assume the image is already a card crop.

    Returns:
        CardIdentity with best match, confidence, and candidates.
    """
    from pokescan.identify.card_db import enrich_candidates, search_tcgdex
    from pokescan.identify.matcher import score_candidates
    from pokescan.identify.ocr import extract_card_text

    # 1. Load image.
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image = cv2.imread(str(image_or_path))
        if image is None:
            return CardIdentity(
                best_match=None,
                confidence=0.0,
                ocr_result=OCRResult(),
            )

    # 2. Crop card from slab (reuse existing card_crop module).
    if not skip_crop:
        try:
            from pokescan.card_crop import crop_image, find_card_crop
            crop_result = find_card_crop(image)
            if crop_result.score > 0.1:
                card_image = crop_image(image, crop_result.bbox, padding=0.02)
            else:
                card_image = image
        except Exception:
            card_image = image
    else:
        card_image = image

    # 3. OCR targeted zones.
    ocr_result = extract_card_text(card_image)

    # Override language if OCR detected one.
    search_lang = ocr_result.language or language

    # 4. Search TCGdex.
    candidates = search_tcgdex(ocr_result, language=search_lang)

    if not candidates:
        return CardIdentity(
            best_match=None,
            confidence=0.0,
            ocr_result=ocr_result,
        )

    # 5. Enrich top candidates with full details (HP, pricing, set info).
    candidates = enrich_candidates(candidates, max_enrich=min(top_k * 2, 10))

    # 6. Visual reranking (Phase 2 — automatic when FAISS index exists).
    visual_scores: dict[str, float] | None = None
    try:
        from pokescan.identify.embeddings import is_index_available, visual_search
        if is_index_available():
            vis_results = visual_search(card_image, top_k=50)
            if vis_results:
                visual_scores = {cid: score for cid, score in vis_results}
                # Also inject visual-only candidates that OCR might have missed.
                existing_ids = {c.card_id for c in candidates}
                from pokescan.identify.card_db import enrich_candidate
                from pokescan.identify.models import CardCandidate
                for cid, vscore in vis_results[:5]:
                    if cid not in existing_ids:
                        # Create a minimal candidate from visual match.
                        vis_candidate = CardCandidate(
                            card_id=cid, name="", set_name="", set_id="",
                            number="", number_total=None, language=search_lang,
                        )
                        vis_candidate = enrich_candidate(vis_candidate)
                        candidates.append(vis_candidate)
                        existing_ids.add(cid)
    except ImportError:
        pass  # sentence-transformers or faiss not installed — OCR-only mode.

    # 7. Score and rank.
    scored = score_candidates(ocr_result, candidates, visual_scores=visual_scores)

    # 8. Build result.
    top = scored[:top_k]
    best = top[0] if top else None

    return CardIdentity(
        best_match=best,
        confidence=best.score if best else 0.0,
        ocr_result=ocr_result,
        candidates=top,
    )


def identify_card_from_bytes(
    image_bytes: bytes,
    *,
    language: str = "fr",
    top_k: int = 5,
    skip_crop: bool = False,
) -> CardIdentity:
    """Convenience wrapper for API usage — accepts raw image bytes."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return CardIdentity(
            best_match=None,
            confidence=0.0,
            ocr_result=OCRResult(),
        )
    return identify_card(image, language=language, top_k=top_k, skip_crop=skip_crop)

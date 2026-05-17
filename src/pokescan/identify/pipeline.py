"""Card identification pipeline: crop, OCR, database search, score."""

from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np

from pokescan.identify.models import CardIdentity, OCRResult


def _looks_like_card_crop(image: np.ndarray) -> bool:
    """Return True when the input already looks like a full card crop."""
    h, w = image.shape[:2]
    if h <= 0:
        return False
    ratio = w / h
    return 0.62 <= ratio <= 0.80


def _resize_for_identification(image: np.ndarray) -> tuple[np.ndarray, str]:
    """Downscale very large phone photos before OCR/CLIP."""
    max_dim = int(os.environ.get("POKESCAN_IDENTIFY_MAX_DIM", "1400"))
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, "original"

    scale = max_dim / longest
    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, f"resized:{w}x{h}->{resized.shape[1]}x{resized.shape[0]}"


def _phone_fallback_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Fallback crop for phone photos of raw cards, not graded slabs."""
    h, w = image.shape[:2]
    card_width = int(w * 0.80)
    card_height = int(card_width / (2.5 / 3.5))
    if card_height > h * 0.72:
        card_height = int(h * 0.72)
        card_width = int(card_height * (2.5 / 3.5))
    x = max(0, int((w - card_width) / 2))
    y = max(0, int(h * 0.20))
    if y + card_height > h:
        y = max(0, h - card_height)
    return x, y, card_width, card_height


def identify_card(
    image_or_path: Path | str | np.ndarray,
    *,
    language: str = "fr",
    top_k: int = 5,
    skip_crop: bool = False,
    visual_mode: str = "off",
) -> CardIdentity:
    """Run the full identification pipeline on a card image.

    visual_mode can be "off", "auto", or "always". The default is OCR-only
    because CLIP is too noisy as a fallback for photographed cards.
    """
    from pokescan.identify.card_db import enrich_candidates, search_tcgdex
    from pokescan.identify.matcher import score_candidates
    from pokescan.identify.ocr import extract_card_text

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
    original_shape = image.shape[:2]

    crop_debug = "skipped"
    if not skip_crop and _looks_like_card_crop(image):
        card_image = image
        crop_debug = "already_card_like"
    elif not skip_crop:
        try:
            from pokescan.card_crop import crop_image, find_card_crop

            crop_result = find_card_crop(image)
            if crop_result.score > 0.1:
                card_image = crop_image(image, crop_result.bbox, padding=0.02)
                crop_debug = f"{crop_result.method}:{crop_result.score:.3f}:{crop_result.bbox}"
            elif crop_result.method == "fallback" and not _looks_like_card_crop(image):
                bbox = _phone_fallback_bbox(image)
                card_image = crop_image(image, bbox, padding=0.0)
                crop_debug = f"phone_fallback:{bbox}"
            else:
                card_image = image
                crop_debug = f"kept_original:{crop_result.method}:{crop_result.score:.3f}"
        except Exception:
            card_image = image
            crop_debug = "crop_error_kept_original"
    else:
        card_image = image

    card_image, resize_debug = _resize_for_identification(card_image)
    ocr_result = extract_card_text(card_image)
    ocr_result.raw_texts["_crop"] = crop_debug
    ocr_result.raw_texts["_resize"] = resize_debug
    ocr_result.raw_texts["_image_shape"] = f"{original_shape}->{card_image.shape[:2]}"

    search_lang = ocr_result.language or language or "fr"
    visual_mode = visual_mode if visual_mode in {"auto", "always", "off"} else "off"

    candidates = search_tcgdex(ocr_result, language=search_lang)
    has_ocr_candidates = bool(candidates)

    if candidates:
        candidates = enrich_candidates(candidates, max_enrich=min(top_k * 2, 10))
        if not ocr_result.local_id and ocr_result.hp:
            hp_matches = [
                candidate for candidate in candidates
                if candidate.hp is not None and str(candidate.hp) == ocr_result.hp
            ]
            if len(hp_matches) == 1 and hp_matches[0].number:
                match = hp_matches[0]
                ocr_result.collector_number = (
                    f"{match.number}/{match.number_total}" if match.number_total else match.number
                )
                ocr_result.raw_texts["_number_inferred"] = "unique_name_hp_match"
        ocr_scored = score_candidates(ocr_result, candidates)
        best_ocr = ocr_scored[0] if ocr_scored else None
        strong_ocr = (
            visual_mode != "always"
            and bool(ocr_result.name)
            and best_ocr is not None
            and (
                (bool(ocr_result.local_id) and best_ocr.score >= 0.78)
                or (bool(ocr_result.hp) and best_ocr.score >= 0.55)
            )
        )
        if visual_mode == "off" or strong_ocr:
            top = ocr_scored[:top_k]
            best = top[0] if top else None
            if (
                best
                and not ocr_result.collector_number
                and best.score >= 0.55
                and best.number
            ):
                ocr_result.collector_number = (
                    f"{best.number}/{best.number_total}" if best.number_total else best.number
                )
                ocr_result.raw_texts["_number_inferred"] = "best_candidate"
            return CardIdentity(
                best_match=best,
                confidence=best.score if best else 0.0,
                ocr_result=ocr_result,
                candidates=top,
            )

    visual_scores: dict[str, float] | None = None
    should_run_visual = visual_mode == "always" or (visual_mode == "auto" and has_ocr_candidates)
    if should_run_visual:
        try:
            from pokescan.identify.embeddings import is_index_available, visual_search_detailed

            if is_index_available():
                vis_results = visual_search_detailed(card_image, top_k=50)
                if vis_results:
                    visual_scores = {cid: score for cid, score, _lang in vis_results}
                    existing_ids = {c.card_id for c in candidates}

                    from pokescan.identify.card_db import enrich_candidate
                    from pokescan.identify.models import CardCandidate

                    for cid, _vscore, _indexed_lang in vis_results[:5]:
                        if cid in existing_ids:
                            continue
                        vis_candidate = CardCandidate(
                            card_id=cid,
                            name="",
                            set_name="",
                            set_id="",
                            number="",
                            number_total=None,
                            language=search_lang,
                        )
                        vis_candidate = enrich_candidate(vis_candidate)
                        if vis_candidate.language != search_lang:
                            continue
                        candidates.append(vis_candidate)
                        existing_ids.add(cid)
        except Exception as exc:
            ocr_result.raw_texts["_visual_error"] = f"{type(exc).__name__}: {exc}"

    if not candidates:
        return CardIdentity(
            best_match=None,
            confidence=0.0,
            ocr_result=ocr_result,
        )

    scoring_ocr = ocr_result
    if not has_ocr_candidates and visual_scores:
        scoring_ocr = OCRResult(raw_texts=ocr_result.raw_texts)

    scored = score_candidates(scoring_ocr, candidates, visual_scores=visual_scores)
    top = scored[:top_k]
    best = top[0] if top else None
    if (
        best
        and has_ocr_candidates
        and not ocr_result.collector_number
        and best.score >= 0.55
        and best.number
    ):
        ocr_result.collector_number = (
            f"{best.number}/{best.number_total}" if best.number_total else best.number
        )
        ocr_result.raw_texts["_number_inferred"] = "best_candidate"

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
    visual_mode: str = "off",
) -> CardIdentity:
    """Convenience wrapper for API usage: accepts raw image bytes."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return CardIdentity(
            best_match=None,
            confidence=0.0,
            ocr_result=OCRResult(),
        )
    return identify_card(
        image,
        language=language,
        top_k=top_k,
        skip_crop=skip_crop,
        visual_mode=visual_mode,
    )

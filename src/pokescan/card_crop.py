"""Heuristics for cropping graded Pokemon cards out of slab photos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CARD_ASPECT_RATIO = 2.5 / 3.5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class CropCandidate:
    bbox: tuple[int, int, int, int]
    score: float
    method: str


@dataclass(frozen=True)
class CropResult:
    source: Path
    output: Path
    bbox: tuple[int, int, int, int]
    score: float
    method: str


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_image_path(path))


def find_card_crop(image) -> CropCandidate:
    """Return the best card-like crop below the grading label."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    candidates: list[CropCandidate] = []
    candidates.extend(_saturation_candidates(image))
    candidates.extend(_edge_candidates(image))

    if candidates:
        best = max(candidates, key=lambda item: item.score)
        if best.score >= 0.15:
            return best

    fallback = _fallback_bbox(width, height)
    return CropCandidate(bbox=fallback, score=0.0, method="fallback")


def crop_image(image, bbox: tuple[int, int, int, int], padding: float = 0.0):
    x, y, width, height = _expand_bbox(bbox, image.shape[1], image.shape[0], padding)
    return image[y : y + height, x : x + width]


def tighten_card_crop(image, inset: float = 0.0):
    """Crop inside a first-pass card crop to reduce slab holders and plastic edges."""
    if inset > 0:
        return crop_image(image, _inset_bbox(image.shape[1], image.shape[0], inset), padding=0.0), None

    candidate = _inner_card_candidate(image)
    if candidate is None:
        return image, None
    return crop_image(image, candidate.bbox, padding=0.0), candidate


def draw_preview(image, bbox: tuple[int, int, int, int], label: str):
    import cv2

    preview = image.copy()
    x, y, width, height = bbox
    cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 4)
    cv2.putText(
        preview,
        label,
        (max(8, x), max(32, y - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return preview


def _inner_card_candidate(image):
    import cv2

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=1,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = width * height
    candidates: list[CropCandidate] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if x < width * 0.02 or y < height * 0.02:
            continue
        if x + box_width > width * 0.98 or y + box_height > height * 0.98:
            continue
        area = box_width * box_height
        ratio = box_width / box_height if box_height else 0.0
        if area < image_area * 0.70 or area > image_area * 0.98:
            continue
        if not 0.50 <= ratio <= 0.82:
            continue

        ratio_penalty = abs(ratio - CARD_ASPECT_RATIO) * 0.45
        area_score = area / image_area
        center_penalty = abs((x + box_width / 2) - width / 2) / width * 0.12
        score = area_score - ratio_penalty - center_penalty
        candidates.append(CropCandidate((x, y, box_width, box_height), score, "inner_edges"))

    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.score)
    return best if best.score >= 0.60 else None


def _inset_bbox(image_width: int, image_height: int, inset: float) -> tuple[int, int, int, int]:
    inset = max(0.0, min(inset, 0.20))
    x = int(round(image_width * inset))
    y = int(round(image_height * inset))
    width = max(1, image_width - 2 * x)
    height = max(1, image_height - 2 * y)
    return x, y, width, height


def _saturation_candidates(image) -> list[CropCandidate]:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    y_start = int(height * 0.23)
    roi = image[y_start:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = np.where((hsv[:, :, 1] > 35) & (gray > 35), 255, 0).astype("uint8")
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        iterations=1,
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _score_contours(contours, width, height, y_start, "saturation")


def _edge_candidates(image) -> list[CropCandidate]:
    import cv2

    height, width = image.shape[:2]
    y_start = int(height * 0.22)
    roi = image[y_start:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _score_contours(contours, width, height, y_start, "edges")


def _score_contours(contours, image_width: int, image_height: int, y_offset: int, method: str):
    import cv2

    candidates: list[CropCandidate] = []
    image_area = image_width * image_height
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        y += y_offset
        x, y, width, height = _lock_card_aspect(x, y, width, height, image_width, image_height)
        area = width * height
        ratio = width / height if height else 0.0

        if area < image_area * 0.04 or area > image_area * 0.62:
            continue
        if height < image_height * 0.58:
            continue
        if not 0.50 <= ratio <= 0.82:
            continue
        if y < image_height * 0.24:
            continue

        touches_frame = x < 8 or x + width > image_width - 8 or y + height > image_height - 8
        if touches_frame:
            continue

        area_score = area / image_area
        ratio_penalty = abs(ratio - CARD_ASPECT_RATIO) * 0.55
        center_penalty = abs((x + width / 2) - image_width / 2) / image_width * 0.10
        score = area_score - ratio_penalty - center_penalty
        candidates.append(CropCandidate((x, y, width, height), score, f"{method}_ratio"))
    return candidates


def _lock_card_aspect(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Force a card-shaped bbox using the detected width as the anchor."""
    original_width = width
    target_height = int(round(width / CARD_ASPECT_RATIO))

    if target_height > image_height - y:
        target_height = image_height - y
        width = int(round(target_height * CARD_ASPECT_RATIO))
        x += int((original_width - width) / 2)

    if target_height >= height:
        height = target_height
    else:
        target_width = int(round(height * CARD_ASPECT_RATIO))
        if target_width <= image_width:
            x += int((width - target_width) / 2)
            width = target_width

    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))
    return x, y, width, height


def _fallback_bbox(image_width: int, image_height: int) -> tuple[int, int, int, int]:
    card_width = int(image_width * 0.80)
    card_height = int(card_width / CARD_ASPECT_RATIO)
    max_height = int(image_height * 0.72)
    if card_height > max_height:
        card_height = max_height
        card_width = int(card_height * CARD_ASPECT_RATIO)
    x = int((image_width - card_width) / 2)
    y = int(image_height * 0.27)
    if y + card_height > image_height:
        y = max(0, image_height - card_height - int(image_height * 0.04))
    return x, y, card_width, card_height


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    return left, top, right - left, bottom - top

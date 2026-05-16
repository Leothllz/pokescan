#!/usr/bin/env python3
"""Prepare label-free card crops from scraped graded-card images."""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatch
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.card_crop import crop_image, draw_preview, find_card_crop, iter_images, tighten_card_crop
from pokescan.paths import PROJECT_ROOT, SCRAPED_DATASET_DIR


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset_crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop card regions from dataset_pokemon while removing PSA/PCA labels."
    )
    parser.add_argument("--input", type=Path, default=SCRAPED_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Maximum images to process; 0 means all.")
    parser.add_argument(
        "--pattern",
        default="",
        help="Optional glob or substring filter applied to the path relative to input.",
    )
    parser.add_argument("--padding", type=float, default=0.01, help="Crop padding as bbox fraction.")
    parser.add_argument("--tighten", action="store_true", help="Run a second pass on the crop to remove slab edges.")
    parser.add_argument(
        "--tight-inset",
        type=float,
        default=0.0,
        help="When --tighten is enabled, remove this fixed fraction on each crop side.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing crops.")
    parser.add_argument("--previews", action="store_true", help="Save bbox previews for visual QA.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing images.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.exists():
        print(f"Erreur : dossier introuvable: {input_dir}")
        return 1

    images = iter_images(input_dir)
    if args.pattern:
        pattern = args.pattern.replace("\\", "/")
        images = [
            image
            for image in images
            if _matches_pattern(image.relative_to(input_dir), pattern)
        ]
    if args.limit > 0:
        images = images[: args.limit]

    if args.dry_run:
        print(f"{len(images)} image(s) seraient preparees depuis {input_dir}")
        return 0

    import cv2

    manifest_path = output_dir / "manifest.csv"
    previews_dir = output_dir / "_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.previews:
        previews_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    processed = 0
    skipped = 0
    failed = 0

    for source in images:
        relative = source.relative_to(input_dir)
        crop_path = output_dir / relative
        preview_path = previews_dir / relative if args.previews else None

        if crop_path.exists() and not args.overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(source))
        if image is None:
            failed += 1
            print(f"Image illisible: {source}")
            continue

        candidate = find_card_crop(image)
        crop = crop_image(image, candidate.bbox, padding=args.padding)
        inner = None
        if args.tighten:
            crop, inner = tighten_card_crop(crop, inset=args.tight_inset)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            failed += 1
            print(f"Ecriture impossible: {crop_path}")
            continue

        if preview_path is not None:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            label = f"{candidate.method} {candidate.score:.3f}"
            if inner is not None:
                label += f" + {inner.method} {inner.score:.3f}"
            preview = draw_preview(image, candidate.bbox, label)
            cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90])

        grade = relative.parts[0] if len(relative.parts) > 0 else ""
        side = relative.parts[1] if len(relative.parts) > 1 else ""
        x, y, width, height = candidate.bbox
        rows.append(
            {
                "source": str(source),
                "crop": str(crop_path),
                "grade": grade,
                "side": side,
                "method": candidate.method,
                "inner_method": inner.method if inner is not None else "",
                "inner_score": f"{inner.score:.6f}" if inner is not None else "",
                "score": f"{candidate.score:.6f}",
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        )
        processed += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "crop",
                "grade",
                "side",
                "method",
                "inner_method",
                "score",
                "inner_score",
                "x",
                "y",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Crops crees: {processed}")
    print(f"Images ignorees: {skipped}")
    print(f"Echecs: {failed}")
    print(f"Sortie: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if args.previews:
        print(f"Previews: {previews_dir}")
    return 0 if failed == 0 else 2


def _matches_pattern(relative_path: Path, pattern: str) -> bool:
    value = relative_path.as_posix()
    return pattern in value or fnmatch(value, pattern)


if __name__ == "__main__":
    raise SystemExit(main())

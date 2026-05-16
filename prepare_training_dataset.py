#!/usr/bin/env python3
"""Build a normalized classification dataset from scraped graded-card images."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.card_crop import crop_image, find_card_crop, iter_images, tighten_card_crop
from pokescan.paths import PROJECT_ROOT, SCRAPED_DATASET_DIR


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset_normalized"
DEFAULT_IMAGE_WIDTH = 640
DEFAULT_IMAGE_HEIGHT = 896
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a normalized train/val/test dataset for grade classification."
    )
    parser.add_argument("--input", type=Path, default=SCRAPED_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--img-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--img-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.01)
    parser.add_argument("--tighten", action="store_true")
    parser.add_argument("--tight-inset", type=float, default=0.01)
    parser.add_argument("--include-unknown", action="store_true")
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=PROJECT_ROOT / "dataset_exclude.txt",
        help="Text file of relative input paths to skip.",
    )
    parser.add_argument(
        "--grades",
        default="",
        help="Comma-separated grades to keep, for example: 8,9,10.",
    )
    parser.add_argument(
        "--map-grade",
        action="append",
        default=[],
        help="Map an input grade to another label, for example: --map-grade 9.5=9.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Maximum input images to process; 0 means all.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.exists():
        print(f"Erreur : dossier introuvable: {input_dir}")
        return 1

    images = iter_images(input_dir)
    grade_map = _parse_grade_map(args.map_grade)
    allowed_grades = _parse_grades(args.grades)
    excluded = _read_excludes(args.exclude_file)
    if excluded:
        images = [
            image
            for image in images
            if image.relative_to(input_dir).as_posix() not in excluded
        ]
    if not args.include_unknown:
        images = [image for image in images if _grade_from_path(input_dir, image).lower() != "unknown"]
    if allowed_grades:
        images = [
            image
            for image in images
            if _mapped_grade(input_dir, image, grade_map) in allowed_grades
        ]
    if args.limit > 0:
        images = images[: args.limit]

    if args.dry_run:
        print(f"{len(images)} image(s) seraient preparees depuis {input_dir}")
        return 0

    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rows = []
    counts = {split: {} for split in SPLITS}
    processed = 0
    skipped = 0
    failed = 0

    for source in images:
        relative = source.relative_to(input_dir)
        source_grade = _grade_from_path(input_dir, source)
        grade = grade_map.get(source_grade, source_grade)
        side = relative.parts[1] if len(relative.parts) > 1 else "other"
        split = _split_for(relative.as_posix(), args.seed, args.val_ratio, args.test_ratio)
        output_name = _output_name(relative)
        destination = output_dir / split / grade / output_name

        if destination.exists() and not args.overwrite:
            skipped += 1
            continue

        image = cv2.imread(str(source))
        if image is None:
            failed += 1
            print(f"Image illisible: {source}")
            continue

        candidate = find_card_crop(image)
        crop = crop_image(image, candidate.bbox, padding=args.padding)
        inner_method = ""
        inner_score = ""
        if args.tighten:
            crop, inner = tighten_card_crop(crop, inset=args.tight_inset)
            if inner is not None:
                inner_method = inner.method
                inner_score = f"{inner.score:.6f}"

        normalized = cv2.resize(
            crop,
            (args.img_width, args.img_height),
            interpolation=cv2.INTER_AREA,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(destination), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            failed += 1
            print(f"Ecriture impossible: {destination}")
            continue

        counts[split][grade] = counts[split].get(grade, 0) + 1
        x, y, width, height = candidate.bbox
        rows.append(
            {
                "source": str(source),
                "output": str(destination),
                "grade": grade,
                "source_grade": source_grade,
                "side": side,
                "split": split,
                "method": candidate.method,
                "score": f"{candidate.score:.6f}",
                "inner_method": inner_method,
                "inner_score": inner_score,
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
                "output",
                "grade",
                "source_grade",
                "side",
                "split",
                "method",
                "score",
                "inner_method",
                "inner_score",
                "x",
                "y",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images preparees: {processed}")
    print(f"Images ignorees: {skipped}")
    print(f"Echecs: {failed}")
    print(f"Sortie: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print(_format_counts(counts))
    return 0 if failed == 0 else 2


def _grade_from_path(input_dir: Path, source: Path) -> str:
    relative = source.relative_to(input_dir)
    return relative.parts[0] if relative.parts else "unknown"


def _mapped_grade(input_dir: Path, source: Path, grade_map: dict[str, str]) -> str:
    grade = _grade_from_path(input_dir, source)
    return grade_map.get(grade, grade)


def _parse_grades(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _parse_grade_map(values: list[str]) -> dict[str, str]:
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --map-grade value: {value!r}; expected FROM=TO")
        source, target = value.split("=", 1)
        source = source.strip()
        target = target.strip()
        if not source or not target:
            raise ValueError(f"Invalid --map-grade value: {value!r}; expected FROM=TO")
        mapping[source] = target
    return mapping


def _read_excludes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _split_for(value: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def _output_name(relative: Path) -> str:
    parts = list(relative.parts)
    if len(parts) >= 3:
        return f"{parts[1]}_{parts[-1]}"
    return relative.name


def _format_counts(counts: dict[str, dict[str, int]]) -> str:
    grades = sorted({grade for split_counts in counts.values() for grade in split_counts})
    lines = [f"{'grade':<10} {'train':>7} {'val':>7} {'test':>7} {'total':>7}", "-" * 42]
    for grade in grades:
        train = counts["train"].get(grade, 0)
        val = counts["val"].get(grade, 0)
        test = counts["test"].get(grade, 0)
        lines.append(f"{grade:<10} {train:>7} {val:>7} {test:>7} {train + val + test:>7}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())

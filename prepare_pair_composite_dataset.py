#!/usr/bin/env python3
"""Create one YOLO classification image per paired front/back card."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "dataset_pairs_high_grade"
DEFAULT_OUTPUT = ROOT / "dataset_pair_composite_high_grade"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build square front/back composites for YOLO classification.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=1280)
    parser.add_argument("--gap", type=int, default=32)
    parser.add_argument("--margin", type=int, default=24)
    parser.add_argument("--background", type=int, default=28)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    manifest_path = input_dir / "manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest introuvable: {manifest_path}")
        return 1

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    rows = read_manifest(manifest_path)
    output_rows = []
    counts = {"train": {}, "val": {}, "test": {}}
    failed = 0

    for row in rows:
        split = row["split"]
        grade = row["grade"]
        card_id = row["card_id"]
        destination = output_dir / split / grade / f"{card_id}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)

        front = cv2.imread(row["front_image"])
        back = cv2.imread(row["back_image"])
        if front is None or back is None:
            print(f"Paire illisible: {card_id}")
            failed += 1
            continue

        composite = compose_pair(
            front,
            back,
            size=args.size,
            margin=args.margin,
            gap=args.gap,
            background=args.background,
            cv2=cv2,
            np=np,
        )
        ok = cv2.imwrite(str(destination), composite, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if not ok:
            print(f"Ecriture impossible: {destination}")
            failed += 1
            continue

        counts[split][grade] = counts[split].get(grade, 0) + 1
        output_rows.append({**row, "composite_image": str(destination)})

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(rows[0].keys()) + ["composite_image"] if rows else ["composite_image"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Composites prepares: {len(output_rows)}")
    print(f"Echecs: {failed}")
    print(f"Sortie: {output_dir}")
    print()
    print(format_counts(counts))
    return 0 if failed == 0 else 2


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def compose_pair(front, back, size: int, margin: int, gap: int, background: int, cv2, np):
    canvas = np.full((size, size, 3), background, dtype=np.uint8)
    slot_width = (size - 2 * margin - gap) // 2
    slot_height = size - 2 * margin

    front_resized = fit_to_slot(front, slot_width, slot_height, cv2)
    back_resized = fit_to_slot(back, slot_width, slot_height, cv2)

    paste_centered(canvas, front_resized, margin, margin, slot_width, slot_height)
    paste_centered(canvas, back_resized, margin + slot_width + gap, margin, slot_width, slot_height)
    return canvas


def fit_to_slot(image, slot_width: int, slot_height: int, cv2):
    height, width = image.shape[:2]
    scale = min(slot_width / width, slot_height / height)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, target, interpolation=cv2.INTER_AREA)


def paste_centered(canvas, image, x: int, y: int, slot_width: int, slot_height: int) -> None:
    height, width = image.shape[:2]
    x0 = x + (slot_width - width) // 2
    y0 = y + (slot_height - height) // 2
    canvas[y0 : y0 + height, x0 : x0 + width] = image


def format_counts(counts: dict[str, dict[str, int]]) -> str:
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

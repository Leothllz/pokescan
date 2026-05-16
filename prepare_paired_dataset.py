#!/usr/bin/env python3
"""Build a front/back paired dataset for card-level grade classification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.card_crop import crop_image, find_card_crop, tighten_card_crop
from pokescan.paths import SCRAPED_DATASET_DIR
from prepare_training_dataset import _parse_grade_map, _parse_grades, _read_excludes


DEFAULT_OUTPUT_DIR = ROOT / "dataset_pairs"
IMAGE_RE = re.compile(r"^img_(?P<timestamp>\d+)_(?P<index>\d+)\.(?:jpg|jpeg|png|webp)$", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create paired front/back card dataset.")
    parser.add_argument("--input", type=Path, default=SCRAPED_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grades", default="8,9,10")
    parser.add_argument("--map-grade", action="append", default=[])
    parser.add_argument("--exclude-file", type=Path, default=ROOT / "dataset_exclude.txt")
    parser.add_argument("--img-width", type=int, default=640)
    parser.add_argument("--img-height", type=int, default=896)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.01)
    parser.add_argument("--max-timestamp-gap", type=int, default=10)
    parser.add_argument("--tighten", action="store_true")
    parser.add_argument("--tight-inset", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    allowed_grades = _parse_grades(args.grades)
    grade_map = _parse_grade_map(args.map_grade)
    excludes = _read_excludes(args.exclude_file)

    pairs, skipped = build_pairs(
        input_dir,
        allowed_grades,
        grade_map,
        excludes,
        max_timestamp_gap=args.max_timestamp_gap,
    )
    if args.dry_run:
        print(f"{len(pairs)} paire(s) seraient preparees depuis {input_dir}")
        print_skipped(skipped)
        return 0

    import cv2

    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    rows = []
    counts = {"train": {}, "val": {}, "test": {}}
    processed = 0
    failed = 0

    for pair in pairs:
        split = split_for(pair["card_id"], args.seed, args.val_ratio, args.test_ratio)
        grade = pair["grade"]
        front_out = output_dir / split / grade / f"{pair['card_id']}_front.jpg"
        back_out = output_dir / split / grade / f"{pair['card_id']}_back.jpg"

        if front_out.exists() and back_out.exists() and not args.overwrite:
            continue

        ok_front = process_image(
            pair["front_path"],
            front_out,
            args.img_width,
            args.img_height,
            args.padding,
            args.tighten,
            args.tight_inset,
            cv2,
        )
        ok_back = process_image(
            pair["back_path"],
            back_out,
            args.img_width,
            args.img_height,
            args.padding,
            args.tighten,
            args.tight_inset,
            cv2,
        )
        if not (ok_front and ok_back):
            failed += 1
            continue

        counts[split][grade] = counts[split].get(grade, 0) + 1
        rows.append(
            {
                "card_id": pair["card_id"],
                "grade": grade,
                "source_grade": pair["source_grade"],
                "split": split,
                "front_source": str(pair["front_path"]),
                "back_source": str(pair["back_path"]),
                "front_image": str(front_out),
                "back_image": str(back_out),
            }
        )
        processed += 1

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "card_id",
                "grade",
                "source_grade",
                "split",
                "front_source",
                "back_source",
                "front_image",
                "back_image",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Paires preparees: {processed}")
    print(f"Echecs: {failed}")
    print(f"Sortie: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print(format_counts(counts))
    print()
    print_skipped(skipped)
    return 0 if failed == 0 else 2


def build_pairs(
    input_dir: Path,
    allowed_grades: set[str],
    grade_map: dict[str, str],
    excludes: set[str],
    max_timestamp_gap: int,
) -> tuple[list[dict], dict[str, int]]:
    pairs = []
    skipped = {
        "bad_name": 0,
        "excluded_image": 0,
        "missing_side": 0,
        "ambiguous_back": 0,
    }

    for grade_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        source_grade = grade_dir.name
        grade = grade_map.get(source_grade, source_grade)
        if allowed_grades and grade not in allowed_grades:
            continue

        front_dir = grade_dir / "front"
        back_dir = grade_dir / "back"
        if not front_dir.exists() or not back_dir.exists():
            skipped["missing_side"] += 1
            continue

        backs_by_index = {}
        for back_path in sorted(back_dir.iterdir()):
            parsed = parse_image_name(back_path)
            if parsed is None:
                skipped["bad_name"] += 1
                continue
            if back_path.relative_to(input_dir).as_posix() in excludes:
                skipped["excluded_image"] += 1
                continue
            backs_by_index.setdefault(parsed["index"], []).append((parsed, back_path))

        used_backs = set()
        for front_path in sorted(front_dir.iterdir()):
            parsed = parse_image_name(front_path)
            if parsed is None:
                skipped["bad_name"] += 1
                continue

            front_rel = front_path.relative_to(input_dir).as_posix()
            if front_rel in excludes:
                skipped["excluded_image"] += 1
                continue

            candidates = []
            for target_index in (parsed["index"] + 1, parsed["index"] - 1):
                for back_parsed, path in backs_by_index.get(target_index, []):
                    if path in used_backs:
                        continue
                    timestamp_gap = abs(back_parsed["timestamp"] - parsed["timestamp"])
                    if timestamp_gap > max_timestamp_gap:
                        continue
                    candidates.append((timestamp_gap, path, back_parsed))

            if not candidates:
                skipped["missing_side"] += 1
                continue
            candidates.sort(key=lambda item: item[0])
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                skipped["ambiguous_back"] += 1
                continue

            _, back_path, back_parsed = candidates[0]
            used_backs.add(back_path)
            card_id = f"{source_grade}_{parsed['timestamp']}_{min(parsed['index'], back_parsed['index'])}"
            pairs.append(
                {
                    "card_id": card_id,
                    "grade": grade,
                    "source_grade": source_grade,
                    "front_path": front_path,
                    "back_path": back_path,
                }
            )
    return pairs, skipped


def parse_image_name(path: Path) -> dict[str, int] | None:
    match = IMAGE_RE.match(path.name)
    if not match:
        return None
    return {"timestamp": int(match.group("timestamp")), "index": int(match.group("index"))}


def process_image(
    source: Path,
    destination: Path,
    width: int,
    height: int,
    padding: float,
    tighten: bool,
    tight_inset: float,
    cv2,
) -> bool:
    image = cv2.imread(str(source))
    if image is None:
        print(f"Image illisible: {source}")
        return False
    candidate = find_card_crop(image)
    crop = crop_image(image, candidate.bbox, padding=padding)
    if tighten:
        crop, _ = tighten_card_crop(crop, inset=tight_inset)
    normalized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(destination), normalized, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def split_for(value: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def format_counts(counts: dict[str, dict[str, int]]) -> str:
    grades = sorted({grade for split_counts in counts.values() for grade in split_counts})
    lines = [f"{'grade':<10} {'train':>7} {'val':>7} {'test':>7} {'total':>7}", "-" * 42]
    for grade in grades:
        train = counts["train"].get(grade, 0)
        val = counts["val"].get(grade, 0)
        test = counts["test"].get(grade, 0)
        lines.append(f"{grade:<10} {train:>7} {val:>7} {test:>7} {train + val + test:>7}")
    return "\n".join(lines)


def print_skipped(skipped: dict[str, int]) -> None:
    print("Paires ignorees:")
    for key, value in skipped.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())

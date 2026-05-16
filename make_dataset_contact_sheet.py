#!/usr/bin/env python3
"""Create contact sheets to review normalized training images quickly."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make QA contact sheets from a normalized dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset_high_grade"))
    parser.add_argument("--output", type=Path, default=Path("runs/qa_contact_sheets"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--grade", default="", help="Optional grade folder to sample.")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.dataset.resolve()
    image_root = root / args.split
    if args.grade:
        image_root = image_root / args.grade
    if not image_root.exists():
        print(f"Erreur : dossier introuvable: {image_root}")
        return 1

    image_paths = sorted(
        path
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    random.Random(args.seed).shuffle(image_paths)
    image_paths = image_paths[: args.samples]
    if not image_paths:
        print("Aucune image trouvee.")
        return 1

    import cv2
    import numpy as np

    thumbs = []
    labels = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        thumb_height = int(round(height * args.thumb_width / width))
        thumb = cv2.resize(image, (args.thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        thumbs.append(thumb)
        labels.append(path.relative_to(root).as_posix())

    if not thumbs:
        print("Aucune image lisible.")
        return 1

    label_height = 34
    cell_width = args.thumb_width
    cell_height = max(thumb.shape[0] for thumb in thumbs) + label_height
    cols = max(1, args.cols)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.full((rows * cell_height, cols * cell_width, 3), 245, dtype=np.uint8)

    for index, (thumb, label) in enumerate(zip(thumbs, labels)):
        row = index // cols
        col = index % cols
        y = row * cell_height
        x = col * cell_width
        sheet[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
        cv2.putText(
            sheet,
            label[-28:],
            (x + 4, y + cell_height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_{args.grade or 'all'}"
    output_path = args.output / f"{suffix}.jpg"
    csv_path = args.output / f"{suffix}.csv"
    cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path"])
        for path in image_paths:
            writer.writerow([path.relative_to(root).as_posix()])

    print(f"Contact sheet: {output_path}")
    print(f"Index: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

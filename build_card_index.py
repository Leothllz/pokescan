#!/usr/bin/env python3
"""Build the FAISS visual index from TCGdex card images.

Downloads official card images from TCGdex across multiple languages,
encodes them with CLIP, and builds a FAISS index for visual search.

Cards are merged by card_id: if a card exists in EN, we skip it in JA/FR.
This gives full coverage (including JP-only exclusives) in a single index.

Usage:
    python build_card_index.py
    python build_card_index.py --languages en ja fr --max-per-set 0
    python build_card_index.py --languages en --max-per-set 50  # quick test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TCGDEX_BASE = "https://api.tcgdex.net/v2"


def _fetch_json(url: str) -> list | dict | None:
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


def _download_image(url: str) -> np.ndarray | None:
    """Download an image and return as BGR numpy array."""
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return None
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS visual index from TCGdex card images",
    )
    parser.add_argument(
        "--languages", nargs="+", default=["en", "ja", "fr"],
        help="Languages to index, in priority order (default: en ja fr)",
    )
    parser.add_argument(
        "--max-per-set", type=int, default=0,
        help="Max cards per set (0 = all). Use 10-50 for quick tests.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: data/embeddings/)",
    )
    args = parser.parse_args(argv)

    from pokescan.identify.embeddings import build_index
    from pokescan.paths import DATA_DIR

    output_dir = Path(args.output) if args.output else DATA_DIR / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PokeScan — Build Card Visual Index")
    print("=" * 60)
    print(f"  Languages : {', '.join(args.languages)}")
    print(f"  Max/set   : {'all' if args.max_per_set == 0 else args.max_per_set}")
    print(f"  Output    : {output_dir}")
    print("=" * 60)

    # Collect all card images across languages, deduplicating by card_id.
    seen_ids: set[str] = set()
    card_images: list[tuple[str, np.ndarray]] = []
    total_skipped = 0
    total_failed = 0

    for lang in args.languages:
        print(f"\n--- Language: {lang.upper()} ---")

        # Fetch all sets for this language.
        sets_data = _fetch_json(f"{TCGDEX_BASE}/{lang}/sets")
        if not sets_data or not isinstance(sets_data, list):
            print(f"  Could not fetch sets for {lang}")
            continue

        print(f"  {len(sets_data)} sets found")

        for set_idx, set_info in enumerate(sets_data):
            set_id = set_info.get("id", "")
            set_name = set_info.get("name", set_id)
            card_count = set_info.get("cardCount", {}).get("total", "?")

            # Fetch cards in this set.
            cards_data = _fetch_json(f"{TCGDEX_BASE}/{lang}/sets/{set_id}")
            if not cards_data or not isinstance(cards_data, dict):
                continue

            cards = cards_data.get("cards", [])
            if not cards:
                continue

            processed = 0
            set_new = 0

            for card in cards:
                card_id = card.get("id", "")
                image_url = card.get("image")

                if not card_id or not image_url:
                    continue

                # Deduplicate across languages.
                if card_id in seen_ids:
                    total_skipped += 1
                    continue

                # Download image.
                full_url = image_url + "/high.webp"
                img = _download_image(full_url)
                if img is None:
                    # Try without /high.webp suffix.
                    img = _download_image(image_url + ".webp")
                if img is None:
                    total_failed += 1
                    continue

                seen_ids.add(card_id)
                card_images.append((card_id, img))
                set_new += 1
                processed += 1

                if args.max_per_set > 0 and processed >= args.max_per_set:
                    break

                # Gentle rate limiting.
                if processed % 20 == 0:
                    time.sleep(0.2)

            if set_new > 0:
                progress = f"[{set_idx + 1}/{len(sets_data)}]"
                print(f"  {progress} {set_name}: +{set_new} new (total: {len(card_images)})")

            # Brief pause between sets.
            time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"  Total images collected : {len(card_images)}")
    print(f"  Skipped (duplicates)   : {total_skipped}")
    print(f"  Failed downloads       : {total_failed}")
    print(f"{'=' * 60}")

    if not card_images:
        print("  No images to index!")
        return 1

    # Build the FAISS index.
    print(f"\n  Encoding {len(card_images)} images with CLIP...")
    print("  (This may take a while on CPU — ~1-2 images/sec)")

    start = time.time()
    count = build_index(card_images, output_dir)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  Index built: {count} cards in {elapsed:.0f}s")
    print(f"  Saved to: {output_dir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

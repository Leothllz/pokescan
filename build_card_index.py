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
    parser.add_argument(
        "--checkpoint-every", type=int, default=250,
        help="Write a partial FAISS index every N encoded cards (default: 250).",
    )
    parser.add_argument(
        "--encode-batch-size", type=int, default=32,
        help="Number of downloaded images to encode together on CLIP (default: 32).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing index in the output directory.",
    )
    args = parser.parse_args(argv)
    args.encode_batch_size = max(1, args.encode_batch_size)

    from pokescan.identify.embeddings import (
        encode_image,
        encode_images,
        load_saved_embeddings,
        save_index,
    )
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

    # Encode cards as they are downloaded, deduplicating by card_id.
    # Keep only CLIP vectors in memory, never the full image catalog.
    seen_ids: set[str] = set()
    card_embeddings: list[tuple[str, str, np.ndarray]] = []
    pending_images: list[tuple[str, str, np.ndarray]] = []
    total_skipped = 0
    total_failed = 0
    total_encode_failed = 0

    if args.resume:
        resumed = load_saved_embeddings(output_dir)
        card_embeddings.extend(resumed)
        seen_ids.update(card_id for card_id, _lang, _embedding in resumed)
        if resumed:
            print(f"  Resumed existing index: {len(resumed)} vectors")

    def add_embedding(card_id: str, lang: str, embedding: np.ndarray) -> None:
        card_embeddings.append((card_id, lang, embedding))
        if (
            args.checkpoint_every > 0
            and len(card_embeddings) % args.checkpoint_every == 0
        ):
            save_index(card_embeddings, output_dir)
            print(f"  Checkpoint saved: {len(card_embeddings)} vectors", flush=True)

    def flush_pending() -> int:
        nonlocal total_encode_failed

        if not pending_images:
            return 0

        batch = list(pending_images)
        pending_images.clear()
        print(
            f"  Encoding batch: {batch[0][0]} ... {batch[-1][0]} ({len(batch)})",
            flush=True,
        )

        try:
            vectors = encode_images(
                [image for _card_id, _lang, image in batch],
                batch_size=max(1, args.encode_batch_size),
            )
            for (card_id, lang, _image), embedding in zip(batch, vectors):
                add_embedding(card_id, lang, embedding)
            return len(batch)
        except Exception as exc:
            print(f"  Batch encode failed: {type(exc).__name__}: {exc}")

        encoded = 0
        for card_id, lang, image in batch:
            try:
                add_embedding(card_id, lang, encode_image(image))
                encoded += 1
            except Exception as exc:
                total_encode_failed += 1
                print(f"  Encode failed for {card_id}: {type(exc).__name__}: {exc}")
        return encoded

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
                pending_images.append((card_id, lang, img))
                processed += 1

                if len(pending_images) >= args.encode_batch_size:
                    set_new += flush_pending()

                if args.max_per_set > 0 and processed >= args.max_per_set:
                    break

                # Gentle rate limiting.
                if processed % 20 == 0:
                    time.sleep(0.2)

            set_new += flush_pending()

            if set_new > 0:
                progress = f"[{set_idx + 1}/{len(sets_data)}]"
                print(f"  {progress} {set_name}: +{set_new} new (total: {len(card_embeddings)})")

            # Brief pause between sets.
            time.sleep(0.1)

    print(f"\n{'=' * 60}")
    print(f"  Total cards encoded    : {len(card_embeddings)}")
    print(f"  Skipped (duplicates)   : {total_skipped}")
    print(f"  Failed downloads       : {total_failed}")
    print(f"  Failed encodes         : {total_encode_failed}")
    print(f"{'=' * 60}")

    if not card_embeddings:
        print("  No images to index!")
        return 1

    # Build the FAISS index from precomputed vectors.
    print(f"\n  Writing FAISS index for {len(card_embeddings)} vectors...")

    start = time.time()
    count = save_index(card_embeddings, output_dir)
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print(f"  Index built: {count} cards in {elapsed:.0f}s")
    print(f"  Saved to: {output_dir}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

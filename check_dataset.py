#!/usr/bin/env python3
"""Print statistics for the scraped Pokemon card dataset."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.dataset_stats import collect_grade_stats, format_grade_stats
from pokescan.paths import SCRAPED_DATASET_DIR


def check_stats(dataset_path: Path = SCRAPED_DATASET_DIR) -> int:
    try:
        stats = collect_grade_stats(dataset_path)
    except FileNotFoundError as exc:
        print(f"Erreur : {exc}")
        return 1

    print(format_grade_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(check_stats())

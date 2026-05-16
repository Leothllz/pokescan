from pathlib import Path

import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.dataset_stats import collect_grade_stats, format_grade_stats


def test_collect_grade_stats_counts_front_back_and_other(tmp_path):
    dataset = tmp_path / "dataset_pokemon"
    (dataset / "10" / "front").mkdir(parents=True)
    (dataset / "10" / "back").mkdir(parents=True)
    (dataset / "10" / "front" / "a.jpg").write_bytes(b"x")
    (dataset / "10" / "back" / "b.png").write_bytes(b"x")
    (dataset / "10" / "loose.webp").write_bytes(b"x")
    (dataset / "unknown" / "front").mkdir(parents=True)
    (dataset / "unknown" / "front" / "c.txt").write_text("not an image")

    stats = collect_grade_stats(dataset)

    assert len(stats) == 2
    assert stats[0].grade == "10"
    assert stats[0].front == 1
    assert stats[0].back == 1
    assert stats[0].other == 1
    assert stats[0].total == 3
    assert stats[1].grade == "unknown"
    assert stats[1].total == 0


def test_format_grade_stats_includes_total():
    text = format_grade_stats([])

    assert "STATISTIQUES DU DATASET" in text
    assert "TOTAL" in text

"""Dataset statistics helpers."""

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class GradeStats:
    grade: str
    front: int
    back: int
    other: int

    @property
    def total(self) -> int:
        return self.front + self.back + self.other


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def collect_grade_stats(dataset_dir: Path) -> list[GradeStats]:
    """Return image counts per grade for the scraped dataset layout."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    stats: list[GradeStats] = []
    for grade_dir in sorted((p for p in dataset_dir.iterdir() if p.is_dir()), key=lambda p: p.name):
        front_dir = grade_dir / "front"
        back_dir = grade_dir / "back"
        front = count_images(front_dir)
        back = count_images(back_dir)
        direct_images = sum(
            1
            for item in grade_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        nested_other = count_images(grade_dir) - front - back - direct_images
        other = max(0, direct_images + nested_other)
        stats.append(GradeStats(grade=grade_dir.name, front=front, back=back, other=other))
    return stats


def format_grade_stats(stats: list[GradeStats]) -> str:
    total = sum(item.total for item in stats)
    lines = [
        "=" * 54,
        "STATISTIQUES DU DATASET",
        "=" * 54,
        f"{'Note':<10} {'Front':>7} {'Back':>7} {'Other':>7} {'Total':>7}",
        "-" * 54,
    ]
    for item in stats:
        lines.append(
            f"{item.grade:<10} {item.front:>7} {item.back:>7} {item.other:>7} {item.total:>7}"
        )
    lines.extend(["-" * 54, f"{'TOTAL':<10} {'':>7} {'':>7} {'':>7} {total:>7}", "=" * 54])
    return "\n".join(lines)

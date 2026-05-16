"""Validation helpers for CLI entrypoints."""

from pathlib import Path


def require_file(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path

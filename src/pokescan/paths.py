"""Shared filesystem paths for PokeScan."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRAPED_DATASET_DIR = PROJECT_ROOT / "dataset_pokemon"
YOLO_DATASET_DIR = PROJECT_ROOT / "dataset" / "Card Grader.v1i.yolov11"
YOLO_DATA_YAML = YOLO_DATASET_DIR / "data.yaml"
YOLO_VAL_IMAGE_DIR = YOLO_DATASET_DIR / "valid" / "images"
YOLO_VAL_LABEL_DIR = YOLO_DATASET_DIR / "valid" / "labels"

DATA_DIR = PROJECT_ROOT / "data"
TCGDEX_CACHE = DATA_DIR / "tcgdex_cache.json"

WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_MODEL = WEIGHTS_DIR / "best.pt"
RUNS_DIR = PROJECT_ROOT / "runs"

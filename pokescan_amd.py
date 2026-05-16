#!/usr/bin/env python3
"""PokeScan training, evaluation, and prediction CLI.

This script keeps the original public interface:

    python pokescan_amd.py --mode train
    python pokescan_amd.py --mode eval --model weights/best.pt
    python pokescan_amd.py --mode predict --model weights/best.pt --source image.jpg

The implementation is intentionally defensive: it checks local datasets and
model files before doing expensive work, and it keeps ROCm/AMD setup optional.
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import random
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pokescan.paths import (
    DEFAULT_MODEL,
    RUNS_DIR,
    YOLO_DATASET_DIR,
    YOLO_DATA_YAML,
    YOLO_VAL_IMAGE_DIR,
    YOLO_VAL_LABEL_DIR,
)
from pokescan.validation import require_dir, require_file


# AMD ROCm defaults for RX 6750 XT / gfx1031 spoofing under Linux/WSL2.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("HCC_AMDGPU_TARGET", "gfx1030")
os.environ.setdefault("AMD_LOG_LEVEL", "0")
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")


def print_header() -> None:
    print()
    print("=" * 64)
    print("PokeScan - AI grading for Pokemon cards")
    print("YOLO + AMD RX 6750 XT / ROCm ready")
    print("=" * 64)
    print()


def fail(message: str, exit_code: int = 1) -> None:
    print(f"Erreur : {message}")
    raise SystemExit(exit_code)


def import_torch():
    try:
        import torch
    except ImportError as exc:
        fail(
            "PyTorch n'est pas installe. Installez les dependances avec "
            "`pip install -r requirements.txt`, puis installez la roue ROCm "
            "si vous utilisez WSL2/AMD."
        )
        raise exc
    return torch


def check_amd_gpu(force_cpu: bool = False) -> str:
    """Return an Ultralytics device value: 'cpu' or GPU index '0'."""
    if force_cpu:
        print("Mode CPU force.")
        return "cpu"

    torch = import_torch()

    print("=" * 64)
    print("Verification GPU AMD via ROCm/HIP")
    print("=" * 64)

    if not torch.cuda.is_available():
        print("Aucun GPU detecte via ROCm/HIP. Le script utilisera le CPU.")
        print()
        if platform.system() == "Windows":
            print("Diagnostic Windows :")
            print("  Le spoof gfx1031 -> gfx1030 ne fonctionne pas en Windows natif.")
            print("  Utilisez WSL2/Ubuntu pour l'acceleration ROCm sur RX 6750 XT.")
            print("  Voir setup_amd.md et setup_wsl2_rocm.sh.")
        else:
            print("Verifiez ROCm, PyTorch ROCm et les variables :")
            print("  HSA_OVERRIDE_GFX_VERSION=10.3.0")
            print("  HCC_AMDGPU_TARGET=gfx1030")
            print("  HSA_ENABLE_DXG_DETECTION=1")
        print("=" * 64)
        return "cpu"

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU detecte : {gpu_name}")
    print(f"VRAM        : {gpu_mem:.1f} Go")
    print(f"PyTorch     : {torch.__version__}")
    print("=" * 64)
    return "0"


def clear_gpu_memory() -> None:
    gc.collect()
    try:
        torch = import_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except SystemExit:
        return
    print("Memoire GPU liberee.")


def download_dataset() -> None:
    """Download the Kaggle YOLO dataset only when explicitly called by train."""
    if YOLO_DATASET_DIR.exists() and any(YOLO_DATASET_DIR.iterdir()):
        print(f"Dataset YOLO deja present : {YOLO_DATASET_DIR}")
        return

    print("Telechargement du dataset Card Grader depuis Kaggle...")
    try:
        import kagglehub
    except ImportError:
        fail(
            "kagglehub n'est pas installe. Installez-le ou placez le dataset "
            f"manuellement dans {YOLO_DATASET_DIR}."
        )

    try:
        card_grader_path = Path(kagglehub.dataset_download("adriantseee2/card-grader"))
        YOLO_DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
        src = card_grader_path / "Card Grader.v1i.yolov11"
        shutil.copytree(str(src if src.exists() else card_grader_path), str(YOLO_DATASET_DIR), dirs_exist_ok=True)
        print(f"Dataset copie dans : {YOLO_DATASET_DIR}")
    except Exception as exc:
        fail(f"telechargement du dataset impossible: {exc}")


def ensure_yolo_dataset(allow_download: bool) -> None:
    if not YOLO_DATASET_DIR.exists() or not any(YOLO_DATASET_DIR.iterdir()):
        if allow_download:
            download_dataset()
        else:
            fail(
                "dataset YOLO absent. Placez-le dans "
                f"{YOLO_DATASET_DIR} ou lancez d'abord `python pokescan_amd.py --mode train`."
            )

    require_file(YOLO_DATA_YAML, "Fichier data.yaml")
    require_dir(YOLO_VAL_IMAGE_DIR, "Dossier d'images de validation")
    require_dir(YOLO_VAL_LABEL_DIR, "Dossier de labels de validation")


def fix_data_yaml() -> None:
    """Normalize Kaggle paths in data.yaml for local Ultralytics runs."""
    require_file(YOLO_DATA_YAML, "Fichier data.yaml")

    content = YOLO_DATA_YAML.read_text(encoding="utf-8")
    needs_fix = False

    if "/kaggle/" in content:
        content = content.replace("/kaggle/input/card-grader/Card Grader.v1i.yolov11/", "")
        needs_fix = True

    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            lines.append(f"path: {YOLO_DATASET_DIR}")
            needs_fix = True
        elif stripped.startswith("train:") and "/" in stripped and not stripped.startswith("train: train"):
            lines.append("train: train/images")
            needs_fix = True
        elif stripped.startswith("val:") and "/" in stripped and not stripped.startswith("val: valid"):
            lines.append("val: valid/images")
            needs_fix = True
        elif stripped.startswith("test:") and "/" in stripped and not stripped.startswith("test: test"):
            lines.append("test: test/images")
            needs_fix = True
        else:
            lines.append(line)

    if needs_fix:
        backup = YOLO_DATA_YAML.with_suffix(".yaml.bak")
        if not backup.exists():
            shutil.copy2(YOLO_DATA_YAML, backup)
        YOLO_DATA_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("data.yaml corrige pour l'execution locale.")


def resolve_model_path(model_path: str | None, *, allow_latest_run: bool) -> Path:
    if model_path:
        return require_file(Path(model_path).expanduser(), "Modele")

    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL

    if allow_latest_run:
        detect_dir = RUNS_DIR / "detect"
        if detect_dir.exists():
            runs = sorted((p for p in detect_dir.iterdir() if p.is_dir()), key=os.path.getmtime, reverse=True)
            for run in runs:
                candidate = run / "weights" / "best.pt"
                if candidate.exists():
                    return candidate

    fail(f"modele introuvable. Fournissez --model ou placez un fichier dans {DEFAULT_MODEL}.")
    raise AssertionError("unreachable")


def train_model(device: str, epochs: int, batch_size: int, img_size: int, model_size: str) -> object:
    from ultralytics import YOLO

    ensure_yolo_dataset(allow_download=True)
    fix_data_yaml()

    print()
    print("=" * 64)
    print(f"Entrainement yolo26{model_size}")
    print("=" * 64)
    print(f"Device     : {device}")
    print(f"Epochs     : {epochs}")
    print(f"Batch      : {batch_size}")
    print(f"Image size : {img_size}")
    print(f"Dataset    : {YOLO_DATA_YAML}")
    print("=" * 64)

    model = YOLO(f"yolo26{model_size}.pt")
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(RUNS_DIR / "detect"),
        name=f"card_grader_yolo26{model_size}_{epochs}epochs",
        workers=4,
        amp=True,
        patience=20,
        save=True,
        save_period=10,
        verbose=True,
    )

    print("Entrainement termine.")
    print(
        "Meilleur modele attendu : "
        f"{RUNS_DIR / 'detect' / f'card_grader_yolo26{model_size}_{epochs}epochs' / 'weights' / 'best.pt'}"
    )
    clear_gpu_memory()
    return results


def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, int, int, int, int]]:
    boxes = []
    if not label_path.exists():
        return boxes

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, cx, cy, w, h = map(float, parts[:5])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        boxes.append((int(cls), x1, y1, x2, y2))
    return boxes


def draw_boxes(image, boxes, class_names, is_prediction: bool = False):
    import cv2

    img = image.copy()
    for item in boxes:
        if is_prediction:
            cls, x1, y1, x2, y2, conf = item
            label = f"{class_names[cls]} {conf:.2f}"
            color = (0, 255, 0)
        else:
            cls, x1, y1, x2, y2 = item
            label = f"{class_names[cls]}"
            color = (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def evaluate_model(device: str, model_path: str | None, num_samples: int, save_dir: str | None) -> None:
    ensure_yolo_dataset(allow_download=False)
    model_file = resolve_model_path(model_path, allow_latest_run=True)

    import cv2
    import matplotlib

    matplotlib.use("Agg" if save_dir else "TkAgg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    print(f"Modele : {model_file}")
    model = YOLO(str(model_file))
    class_names = model.names

    image_files = sorted(
        f for f in YOLO_VAL_IMAGE_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        fail(f"aucune image trouvee dans {YOLO_VAL_IMAGE_DIR}")

    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    save_path = Path(save_dir) if save_dir else None
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)

    print(f"Evaluation sur {len(sample_files)} images...")
    for img_path in sample_files:
        label_path = YOLO_VAL_LABEL_DIR / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Impossible de charger : {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]
        gt_boxes = load_ground_truth(label_path, img_w, img_h)
        preds = model(str(img_path), device=device, verbose=False)[0]

        pred_boxes = []
        for box in preds.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pred_boxes.append((int(box.cls[0]), x1, y1, x2, y2, float(box.conf[0])))

        gt_img = cv2.cvtColor(draw_boxes(img, gt_boxes, class_names), cv2.COLOR_BGR2RGB)
        pred_img = cv2.cvtColor(draw_boxes(img, pred_boxes, class_names, True), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(gt_img)
        axes[0].set_title("Ground truth")
        axes[0].axis("off")
        axes[1].imshow(pred_img)
        axes[1].set_title("Predictions")
        axes[1].axis("off")
        plt.suptitle(img_path.name)
        plt.tight_layout()

        if save_path:
            out_path = save_path / f"eval_{img_path.name}"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"Sauvegarde : {out_path}")
            plt.close(fig)
        else:
            plt.show()

    print("Evaluation terminee.")
    clear_gpu_memory()


def predict_images(device: str, model_path: str | None, source: str | None, conf_threshold: float) -> object:
    model_file = resolve_model_path(model_path, allow_latest_run=False)
    if not source:
        fail("source manquante. Utilisez --source avec une image, un dossier ou une URL.")

    source_path = Path(source)
    if "://" not in source and not source_path.exists():
        fail(f"source introuvable: {source}")

    from ultralytics import YOLO

    print(f"Modele : {model_file}")
    print(f"Source : {source}")
    model = YOLO(str(model_file))
    results = model.predict(
        source=source,
        device=device,
        conf=conf_threshold,
        save=True,
        project=str(RUNS_DIR / "predict"),
        name="yolo26_predictions",
        verbose=True,
    )
    print(f"Predictions sauvegardees dans : {RUNS_DIR / 'predict'}")
    clear_gpu_memory()
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PokeScan - AI grading pour cartes Pokemon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python pokescan_amd.py --mode train --epochs 100 --batch 16
  python pokescan_amd.py --mode eval --model weights/best.pt --samples 20 --save-dir runs/eval
  python pokescan_amd.py --mode predict --model weights/best.pt --source image.jpg
        """,
    )
    parser.add_argument("--mode", required=True, choices=["train", "eval", "predict"])
    parser.add_argument("--model", default=None, help="Chemin vers un modele .pt")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--source", default=None, help="Image, dossier ou URL pour predict")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--cpu", action="store_true", help="Forcer l'execution CPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    print_header()
    device = check_amd_gpu(force_cpu=args.cpu)

    try:
        if args.mode == "train":
            train_model(device, args.epochs, args.batch, args.imgsz, args.model_size)
        elif args.mode == "eval":
            evaluate_model(device, args.model, args.samples, args.save_dir)
        elif args.mode == "predict":
            predict_images(device, args.model, args.source, args.conf)
    except FileNotFoundError as exc:
        fail(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

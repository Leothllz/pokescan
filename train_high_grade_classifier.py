#!/usr/bin/env python3
"""Train a first high-grade Pokemon card classifier with Ultralytics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset_high_grade"
DEFAULT_PROJECT = ROOT / "runs" / "classify"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline classifier on grades 8, 9 and 10.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="high_grade_baseline")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP mixed precision.")
    parser.add_argument(
        "--auto-augment",
        default="none",
        choices=["none", "randaugment", "autoaugment", "augmix"],
        help="Classification auto augmentation policy. 'none' is safer for grading defects.",
    )
    parser.add_argument("--erasing", type=float, default=0.0, help="Random erasing probability.")
    parser.add_argument("--fliplr", type=float, default=0.0, help="Horizontal flip probability.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--scale", type=float, default=0.05, help="Random crop scale amount.")
    parser.add_argument("--hsv-h", type=float, default=0.0, help="Hue augmentation fraction.")
    parser.add_argument("--hsv-s", type=float, default=0.0, help="Saturation augmentation fraction.")
    parser.add_argument("--hsv-v", type=float, default=0.0, help="Value augmentation fraction.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data.exists():
        print(f"Erreur : dataset introuvable: {args.data}")
        return 1

    _setup_rocm_env()

    import torch
    from ultralytics import YOLO

    device = "cpu" if args.cpu else args.device
    print(f"torch: {torch.__version__}")
    print(f"hip: {getattr(torch.version, 'hip', None)}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device 0: {torch.cuda.get_device_name(0)}")
    elif not args.cpu:
        print("Aucun GPU visible, bascule en CPU.")
        device = "cpu"

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        pretrained=True,
        amp=args.amp,
        auto_augment=None if args.auto_augment == "none" else args.auto_augment,
        erasing=args.erasing,
        fliplr=args.fliplr,
        flipud=args.flipud,
        scale=args.scale,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        plots=True,
        val=True,
    )
    return 0


def _setup_rocm_env() -> None:
    """Make local Windows ROCm wheel bins and runtime compiler includes visible."""
    nightly = ROOT / "pokescan-amdnightly" / "Lib" / "site-packages"
    legacy = ROOT / "pokescan-rocm312" / "Lib" / "site-packages"

    if nightly.exists():
        core = nightly / "_rocm_sdk_core"
        rocm_bins = [
            core / "bin",
            nightly / "_rocm_sdk_libraries_gfx103X_dgpu" / "bin",
        ]
        includes = _existing_paths(
            [
                Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\include"),
                Path(r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\ucrt"),
                core / "lib" / "llvm" / "lib" / "clang" / "23" / "include",
            ]
        )
        if includes:
            include_value = ";".join(includes)
            os.environ["INCLUDE"] = ";".join([include_value, os.environ.get("INCLUDE", "")])
            os.environ["CPLUS_INCLUDE_PATH"] = include_value
    else:
        rocm_bins = [
            legacy / "_rocm_sdk_coregfx103X-all" / "bin",
            legacy / "_rocm_sdk_develgfx103X-all" / "bin",
            legacy / "_rocm_sdk_libraries_gfx103X_allgfx103X-all" / "bin",
        ]

    existing = [str(path) for path in rocm_bins if path.exists()]
    if existing:
        os.environ["PATH"] = ";".join(existing + [os.environ.get("PATH", "")])
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


def _existing_paths(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths if path.exists()]


if __name__ == "__main__":
    raise SystemExit(main())

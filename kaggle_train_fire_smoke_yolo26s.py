"""
Arrakis-Project: 산림청 특화 모델 (Forest Fire & Smoke Detection)
================================================================
Fine-tune yolo26s.pt on the D-Fire / Smoke-Fire-Detection-YOLO dataset
to detect fire and smoke for Korean Forest Service (산림청) applications.

Dataset: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
Classes: 0 = smoke, 1 = fire
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

CLASS_NAMES = {
    0: "smoke",
    1: "fire",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune yolo26s.pt on D-Fire smoke/fire dataset for 산림청."
    )
    parser.add_argument(
        "--data-root", type=Path,
        help="Root of the fire/smoke dataset with train/val/test splits.",
    )
    parser.add_argument("--weights", type=str, default="yolo26s.pt")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--output-yaml", type=Path,
        default=KAGGLE_WORKING_ROOT / "fire_smoke.yaml",
    )
    parser.add_argument(
        "--project", type=str,
        default=str(KAGGLE_WORKING_ROOT / "runs" / "fire_smoke"),
    )
    parser.add_argument("--name", type=str, default="yolo26s-fire-smoke-p100")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def running_on_kaggle() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def count_image_files(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(
        1 for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def count_nonempty_labels(labels_dir: Path) -> int:
    if not labels_dir.exists():
        return 0
    return sum(1 for p in labels_dir.glob("*.txt") if p.stat().st_size > 0)


def print_split_summary(data_root: Path) -> None:
    print("Dataset summary:")
    for split in ("train", "val", "test"):
        imgs = count_image_files(data_root / split / "images")
        lbls = count_nonempty_labels(data_root / split / "labels")
        print(f"  {split}: {imgs} images, {lbls} labels")


def validate_dataset(data_root: Path) -> None:
    for split in ("train", "val"):
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.exists():
            raise FileNotFoundError(f"Missing: {img_dir}")
        if count_image_files(img_dir) == 0:
            raise RuntimeError(f"No images in {img_dir}")
        if not lbl_dir.exists():
            raise FileNotFoundError(f"Missing: {lbl_dir}")
        if count_nonempty_labels(lbl_dir) == 0:
            raise RuntimeError(f"No labels in {lbl_dir}")


def resolve_data_root(explicit_root: Path | None) -> Path:
    if explicit_root:
        root = explicit_root.resolve()
        validate_dataset(root)
        return root

    if not running_on_kaggle():
        raise FileNotFoundError("Pass --data-root when running outside Kaggle.")

    for candidate in KAGGLE_INPUT_ROOT.rglob("train"):
        parent = candidate.parent
        if (parent / "train" / "images").is_dir() and (parent / "val" / "images").is_dir():
            print(f"Auto-detected dataset root: {parent}")
            validate_dataset(parent)
            return parent.resolve()

    raise FileNotFoundError(
        "Could not find fire/smoke dataset under /kaggle/input.\n"
        "Attach the 'smoke-fire-detection-yolo' dataset to the notebook."
    )


def restructure_if_needed(data_root: Path) -> Path:
    """Handle datasets where structure is images/train instead of train/images."""
    if (data_root / "train" / "images").is_dir():
        return data_root

    if (data_root / "images" / "train").is_dir():
        print("Restructuring dataset: images/train -> train/images ...")
        out = KAGGLE_WORKING_ROOT / "fire_smoke_data"
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            for subdir in ("images", "labels"):
                src = data_root / subdir / split
                dst = out / split / subdir
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        os.symlink(src.resolve(), dst)
        return out

    return data_root


def delete_cache_files(root: Path) -> None:
    for f in root.rglob("*.cache"):
        try:
            f.unlink()
            print(f"Deleted cache: {f}")
        except OSError:
            pass


def write_dataset_yaml(data_root: Path, output_yaml: Path) -> Path:
    output_yaml = output_yaml.resolve()
    payload = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": CLASS_NAMES,
    }
    test_dir = data_root / "test" / "images"
    if test_dir.exists() and count_image_files(test_dir) > 0:
        payload["test"] = "test/images"

    output_yaml.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Dataset YAML written: {output_yaml}")
    return output_yaml


def main() -> None:
    args = parse_args()

    raw_root = resolve_data_root(args.data_root)
    data_root = restructure_if_needed(raw_root)
    validate_dataset(data_root)
    print_split_summary(data_root)

    delete_cache_files(data_root)
    if running_on_kaggle():
        delete_cache_files(KAGGLE_INPUT_ROOT)

    data_yaml = write_dataset_yaml(data_root, args.output_yaml)

    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}, ImgSz: {args.imgsz}, Batch: {args.batch}")

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "patience": args.patience,
        "cache": args.cache,
        "save_period": args.save_period,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "cos_lr": True,
        "close_mosaic": 10,
        "plots": True,
    }

    if args.resume_from:
        resume_path = args.resume_from.resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"Resuming from: {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        model = YOLO(args.weights)
        model.train(pretrained=True, **train_kwargs)

    trainer = getattr(model, "trainer", None)
    if trainer:
        save_dir = Path(trainer.save_dir)
        print(f"\nRun directory: {save_dir}")
        print(f"Best weights:  {save_dir / 'weights' / 'best.pt'}")
        print(f"Last weights:  {save_dir / 'weights' / 'last.pt'}")


if __name__ == "__main__":
    main()

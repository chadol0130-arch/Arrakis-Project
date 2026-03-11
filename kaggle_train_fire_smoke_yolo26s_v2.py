"""
Arrakis-Project: 산림청 특화 모델 v2 (Forest Fire & Smoke Detection)
====================================================================
Fine-tune yolo26s.pt on merged fire/smoke datasets with enhanced
augmentation and higher resolution for improved fire detection.

Datasets:
  1. sayedgamal99/smoke-fire-detection-yolo  (21K+ images, 0=smoke 1=fire)
  2. roscoekerby/firesmoke-detection-yolo-v9  (35K+ images, 0=fire 1=smoke → REMAP)

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

# Datasets that need class ID remapping: source_class -> target_class
# roscoekerby dataset: 0=Fire, 1=Smoke → remap to 0=Smoke(1→0), 1=Fire(0→1)
REMAP_DATASETS = {
    "firesmoke-detection-yolo-v9": {0: 1, 1: 0},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune yolo26s.pt on merged fire/smoke datasets for 산림청 (v2)."
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--weights", type=str, default="yolo26s.pt")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--output-yaml", type=Path,
        default=KAGGLE_WORKING_ROOT / "fire_smoke_v2.yaml",
    )
    parser.add_argument(
        "--project", type=str,
        default=str(KAGGLE_WORKING_ROOT / "runs" / "fire_smoke_v2"),
    )
    parser.add_argument("--name", type=str, default="yolo26s-fire-smoke-v2-p100")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--save-period", type=int, default=10)
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


def print_split_summary(data_root: Path, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Dataset summary:")
    for split in ("train", "val", "test"):
        imgs = count_image_files(data_root / split / "images")
        lbls = count_nonempty_labels(data_root / split / "labels")
        if imgs > 0 or lbls > 0:
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


def find_dataset_roots() -> list[tuple[Path, str]]:
    """Find all valid dataset roots under /kaggle/input."""
    roots = []
    seen = set()
    for candidate in KAGGLE_INPUT_ROOT.rglob("train"):
        parent = candidate.parent
        if parent in seen:
            continue
        seen.add(parent)
        if (parent / "train" / "images").is_dir() and (parent / "val" / "images").is_dir():
            # Also check for images/train structure
            roots.append((parent, parent.name))
        elif (parent / "images" / "train").is_dir():
            roots.append((parent, parent.name))
    return roots


def restructure_if_needed(data_root: Path, name: str) -> Path:
    """Handle datasets where structure is images/train instead of train/images."""
    if (data_root / "train" / "images").is_dir():
        return data_root

    if (data_root / "images" / "train").is_dir():
        print(f"Restructuring {name}: images/train -> train/images ...")
        out = KAGGLE_WORKING_ROOT / f"restructured_{name}"
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


def remap_label_file(src: Path, dst: Path, class_map: dict[int, int]) -> None:
    """Remap class IDs in a YOLO label file."""
    lines = src.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        new_cls = class_map.get(cls_id, cls_id)
        new_lines.append(f"{new_cls} {' '.join(parts[1:])}")
    dst.write_text("\n".join(new_lines) + "\n")


def merge_datasets(
    primary_root: Path,
    additional_roots: list[tuple[Path, str]],
    output_root: Path,
) -> Path:
    """Merge multiple datasets into a single directory with remapped labels."""
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        img_out = output_root / split / "images"
        lbl_out = output_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # Primary dataset: symlink directly (no remap needed)
        src_imgs = primary_root / split / "images"
        src_lbls = primary_root / split / "labels"
        if src_imgs.exists():
            for img in src_imgs.iterdir():
                if img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
                    dst = img_out / img.name
                    if not dst.exists():
                        os.symlink(img.resolve(), dst)
            if src_lbls.exists():
                for lbl in src_lbls.glob("*.txt"):
                    dst = lbl_out / lbl.name
                    if not dst.exists():
                        os.symlink(lbl.resolve(), dst)

        # Additional datasets: copy labels with remap, symlink images
        for ds_root, ds_name in additional_roots:
            class_map = REMAP_DATASETS.get(ds_name, {})
            needs_remap = bool(class_map)

            ds_imgs = ds_root / split / "images"
            ds_lbls = ds_root / split / "labels"

            if not ds_imgs.exists():
                continue

            count = 0
            for img in ds_imgs.iterdir():
                if not (img.is_file() and img.suffix.lower() in IMAGE_SUFFIXES):
                    continue
                # Prefix filename to avoid collisions
                new_name = f"{ds_name}_{img.name}"
                dst_img = img_out / new_name
                if dst_img.exists():
                    continue
                os.symlink(img.resolve(), dst_img)

                # Handle label
                lbl_name = img.stem + ".txt"
                src_lbl = ds_lbls / lbl_name
                dst_lbl = lbl_out / f"{ds_name}_{lbl_name}"
                if src_lbl.exists() and not dst_lbl.exists():
                    if needs_remap:
                        remap_label_file(src_lbl, dst_lbl, class_map)
                    else:
                        os.symlink(src_lbl.resolve(), dst_lbl)
                count += 1

            print(f"  Merged {count} images from {ds_name}/{split}")

    return output_root


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

    if args.data_root:
        # Single dataset mode
        raw_root = args.data_root.resolve()
        data_root = restructure_if_needed(raw_root, "manual")
        validate_dataset(data_root)
    elif running_on_kaggle():
        # Auto-detect and merge all datasets
        ds_roots = find_dataset_roots()
        if not ds_roots:
            raise FileNotFoundError(
                "Could not find any fire/smoke dataset under /kaggle/input.\n"
                "Attach datasets to the notebook."
            )

        print(f"\nFound {len(ds_roots)} dataset(s):")
        structured_roots = []
        for root, name in ds_roots:
            restructured = restructure_if_needed(root, name)
            structured_roots.append((restructured, name))
            print_split_summary(restructured, name)

        if len(structured_roots) == 1:
            data_root = structured_roots[0][0]
        else:
            # Use first dataset as primary, merge the rest
            primary = structured_roots[0]
            additional = structured_roots[1:]
            print(f"\nMerging datasets: primary={primary[1]}, "
                  f"additional={[n for _, n in additional]}")
            merge_out = KAGGLE_WORKING_ROOT / "merged_fire_smoke"
            data_root = merge_datasets(primary[0], additional, merge_out)

        validate_dataset(data_root)
    else:
        raise FileNotFoundError("Pass --data-root when running outside Kaggle.")

    print_split_summary(data_root, "MERGED" if running_on_kaggle() else "")

    delete_cache_files(data_root)
    if running_on_kaggle():
        delete_cache_files(KAGGLE_INPUT_ROOT)

    data_yaml = write_dataset_yaml(data_root, args.output_yaml)

    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Weights: {args.weights}")
    print(f"Epochs: {args.epochs}, ImgSz: {args.imgsz}, Batch: {args.batch}")
    print(f"Enhanced augmentation: mixup=0.15, copy_paste=0.15, "
          f"multi_scale=0.5, cls=1.0")

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
        # Augmentation
        "cos_lr": True,
        "close_mosaic": 15,
        "mixup": 0.15,
        "copy_paste": 0.15,
        "scale": 0.9,
        "multi_scale": 0.5,
        # Loss weights — boost classification loss for better fire/smoke distinction
        "cls": 1.0,
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

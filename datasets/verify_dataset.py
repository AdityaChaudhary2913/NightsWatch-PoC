import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image
from tqdm import tqdm

from utils.env import ensure_dir, get_log_dir
from utils.logger import get_logger


LOGGER = get_logger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def ensure_yolo_dirs(dataset_root: Path) -> dict[str, Path]:
    paths = {
        "images_train": dataset_root / "images" / "train",
        "images_val": dataset_root / "images" / "val",
        "labels_train": dataset_root / "labels" / "train",
        "labels_val": dataset_root / "labels" / "val",
    }
    for path in paths.values():
        ensure_dir(path)
    return paths


def iter_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def yolo_line(class_id: int, box_xyxy: tuple[float, float, float, float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1 or width <= 0 or height <= 0:
        return None
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    values = [cx, cy, bw, bh]
    if any(v < 0.0 or v > 1.0 for v in values):
        return None
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in values)


def write_dataset_yaml(dataset_root: Path, yaml_path: Path, names: list[str]) -> Path:
    ensure_dir(yaml_path.parent)
    payload = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def resolve_dataset_root(path_or_yaml: Path) -> tuple[Path, list[str] | None]:
    path_or_yaml = Path(path_or_yaml)
    if path_or_yaml.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(path_or_yaml.read_text(encoding="utf-8"))
        root = Path(payload.get("path", "."))
        if not root.is_absolute():
            root = (path_or_yaml.parent / root).resolve()
        names = payload.get("names")
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return root, names
    return path_or_yaml, None


def _label_path_for(image_path: Path, dataset_root: Path, split: str) -> Path:
    relative = image_path.relative_to(dataset_root / "images" / split)
    return dataset_root / "labels" / split / relative.with_suffix(".txt")


def _validate_label_file(label_path: Path, num_classes: int | None) -> tuple[int, list[str]]:
    if not label_path.exists():
        return 0, [f"missing label file: {label_path}"]
    errors: list[str] = []
    valid = 0
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_no}: expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_no}: non-numeric YOLO value")
            continue
        if class_id < 0 or (num_classes is not None and class_id >= num_classes):
            errors.append(f"{label_path}:{line_no}: class id {class_id} outside class range")
            continue
        if any(v < 0.0 or v > 1.0 for v in values):
            errors.append(f"{label_path}:{line_no}: normalized box value outside [0, 1]")
            continue
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{label_path}:{line_no}: non-positive width/height")
            continue
        valid += 1
    return valid, errors


def verify_yolo_dataset(
    dataset_path: str | Path,
    names: list[str] | None = None,
    dataset_name: str | None = None,
    source: str | None = None,
    strict: bool = False,
) -> dict:
    dataset_root, yaml_names = resolve_dataset_root(Path(dataset_path))
    class_names = names or yaml_names or []
    num_classes = len(class_names) if class_names else None
    summary = {
        "dataset": dataset_name or dataset_root.name,
        "root": str(dataset_root),
        "source": source or "unknown",
        "num_classes": num_classes or 0,
        "classes": class_names,
        "train_images": 0,
        "val_images": 0,
        "train_labels": 0,
        "val_labels": 0,
        "empty_label_files": 0,
        "invalid_labels": 0,
        "sample_image_size": None,
        "ok": False,
    }
    if not dataset_root.exists():
        if strict:
            raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
        return summary

    errors: list[str] = []
    for split in ("train", "val"):
        images = iter_images(dataset_root / "images" / split)
        summary[f"{split}_images"] = len(images)
        if images and summary["sample_image_size"] is None:
            summary["sample_image_size"] = list(read_image_size(images[0]))
        for image_path in tqdm(images, desc=f"verify {summary['dataset']} {split}", leave=False):
            label_path = _label_path_for(image_path, dataset_root, split)
            valid, label_errors = _validate_label_file(label_path, num_classes)
            summary[f"{split}_labels"] += int(label_path.exists())
            summary["invalid_labels"] += len(label_errors)
            summary["empty_label_files"] += int(valid == 0)
            errors.extend(label_errors[:5])

    summary["ok"] = (
        summary["train_images"] > 0
        and summary["val_images"] > 0
        and summary["invalid_labels"] == 0
    )
    if strict and not summary["ok"]:
        preview = "\n".join(errors[:10])
        raise ValueError(f"Dataset verification failed for {dataset_root}\n{preview}")
    if errors:
        LOGGER.warning("Dataset %s has %d label issues. First issue: %s", dataset_root, len(errors), errors[0])
    return summary


def save_dataset_summaries(summaries: Iterable[dict]) -> Path:
    output_path = get_log_dir() / "dataset_summary.json"
    payload = [dict(item) for item in summaries]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a YOLO-format dataset before training.")
    parser.add_argument("dataset", type=Path, help="Dataset root or Ultralytics YAML path")
    parser.add_argument("--strict", action="store_true", help="Raise on missing data or invalid labels")
    parser.add_argument("--name", default=None, help="Dataset name for summary output")
    args = parser.parse_args()
    summary = verify_yolo_dataset(args.dataset, dataset_name=args.name, strict=args.strict)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml
from tqdm import tqdm

from datasets.prepare_llvip import prepare_llvip
from datasets.verify_dataset import (
    ensure_yolo_dirs,
    iter_images,
    link_or_copy,
    read_image_size,
    verify_yolo_dataset,
    write_dataset_yaml,
    yolo_line,
)
from utils.env import ensure_dir, get_data_dir, get_output_dir
from utils.logger import get_logger, status


LOGGER = get_logger(__name__)
FLIR_KAGGLE_SLUG = "deepnewbie/flir-thermal-images-dataset"
FLIR_CLASSES = ["person", "bicycle", "car", "dog", "other_vehicle"]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    LOGGER.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _prepared(dataset_root: Path) -> bool:
    return bool(iter_images(dataset_root / "images" / "train")) and bool(iter_images(dataset_root / "images" / "val"))


def download_flir_kaggle(raw_dir: Path) -> None:
    ensure_dir(raw_dir)
    if any(raw_dir.rglob("*.json")) and iter_images(raw_dir):
        LOGGER.info("FLIR raw files already present, skipping Kaggle download")
        return
    slug = os.environ.get("FLIR_KAGGLE_SLUG", FLIR_KAGGLE_SLUG)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"])


def download_flir_roboflow(raw_dir: Path) -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set; cannot use Roboflow fallback")
    workspace = os.environ.get("ROBOFLOW_FLIR_WORKSPACE", "self-driving-cars-lfjou")
    project = os.environ.get("ROBOFLOW_FLIR_PROJECT", "flir-camera-objects")
    version = int(os.environ.get("ROBOFLOW_FLIR_VERSION", "2"))
    target = ensure_dir(raw_dir / "roboflow_flir")
    if (target / "data.yaml").exists() and iter_images(target):
        return target
    from roboflow import Roboflow

    LOGGER.info("Downloading FLIR Roboflow fallback: %s/%s v%s", workspace, project, version)
    rf = Roboflow(api_key=api_key)
    dataset = rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(target))
    return Path(dataset.location)


def _split_from_path(path: Path, default: str = "train") -> str:
    text = "/".join(path.parts).lower()
    if "val" in text or "valid" in text or "validation" in text:
        return "val"
    if "test" in text:
        return "val"
    if "train" in text:
        return "train"
    return default


def _normalise_category(name: str) -> int | None:
    value = name.strip().lower().replace(" ", "_").replace("-", "_")
    if value in {"person", "people", "pedestrian"}:
        return 0
    if value in {"bicycle", "bike", "cyclist"}:
        return 1
    if value in {"car", "sedan", "automobile"}:
        return 2
    if value == "dog":
        return 3
    if value in {"truck", "bus", "van", "motorcycle", "motorbike", "train", "other_vehicle", "vehicle"}:
        return 4
    return None


def _index_images(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for image in iter_images(raw_dir):
        index[image.name].append(image)
        index[image.stem].append(image)
        try:
            relative = image.relative_to(raw_dir).as_posix()
            index[relative].append(image)
        except ValueError:
            pass
    return index


def _find_image(image_info: dict, image_index: dict[str, list[Path]]) -> Path | None:
    file_name = str(image_info.get("file_name") or image_info.get("path") or "")
    candidates = []
    for key in (file_name, Path(file_name).name, Path(file_name).stem):
        candidates.extend(image_index.get(key, []))
    if not candidates:
        return None
    split_hint = _split_from_path(Path(file_name), default="")
    if split_hint:
        for candidate in candidates:
            if split_hint in "/".join(candidate.parts).lower():
                return candidate
    return candidates[0]


def convert_flir_coco(raw_dir: Path, dataset_root: Path) -> dict:
    if _prepared(dataset_root):
        LOGGER.info("FLIR YOLO dataset already prepared")
    else:
        ensure_yolo_dirs(dataset_root)
        annotation_files = sorted(p for p in raw_dir.rglob("*.json") if "annotation" in p.name.lower() or "coco" in p.name.lower())
        if not annotation_files:
            annotation_files = sorted(raw_dir.rglob("*.json"))
        if not annotation_files:
            raise FileNotFoundError(f"No FLIR COCO annotation JSON found under {raw_dir}")
        image_index = _index_images(raw_dir)
        converted = 0
        for annotation_path in annotation_files:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            if not {"images", "annotations", "categories"}.issubset(payload):
                continue
            category_lookup = {}
            for category in payload["categories"]:
                mapped = _normalise_category(str(category.get("name", "")))
                if mapped is not None:
                    category_lookup[int(category["id"])] = mapped
            annotations_by_image: dict[int, list[dict]] = defaultdict(list)
            for ann in payload["annotations"]:
                if ann.get("iscrowd", 0):
                    continue
                annotations_by_image[int(ann["image_id"])].append(ann)
            default_split = _split_from_path(annotation_path, default="train")
            for image_info in tqdm(payload["images"], desc=f"convert {annotation_path.name}"):
                image_path = _find_image(image_info, image_index)
                if image_path is None:
                    continue
                split = _split_from_path(image_path, default=default_split)
                if split not in {"train", "val"}:
                    split = default_split
                width = int(image_info.get("width") or 0)
                height = int(image_info.get("height") or 0)
                if width <= 0 or height <= 0:
                    width, height = read_image_size(image_path)
                lines = []
                for ann in annotations_by_image.get(int(image_info["id"]), []):
                    class_id = category_lookup.get(int(ann["category_id"]))
                    if class_id is None:
                        continue
                    x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                    line = yolo_line(class_id, (x, y, x + w, y + h), width, height)
                    if line:
                        lines.append(line)
                if not lines:
                    continue
                dst_image = dataset_root / "images" / split / image_path.name
                dst_label = dataset_root / "labels" / split / f"{image_path.stem}.txt"
                link_or_copy(image_path, dst_image)
                dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
                converted += 1
        if converted == 0:
            raise RuntimeError("FLIR conversion produced zero labelled images")
    config_dir = ensure_dir(get_output_dir() / "dataset_configs")
    yaml_path = write_dataset_yaml(dataset_root, config_dir / "flir_thermal.yaml", FLIR_CLASSES)
    summary = verify_yolo_dataset(dataset_root, names=FLIR_CLASSES, dataset_name="FLIR thermal", source="FLIR ADAS")
    return {"root": str(dataset_root), "yaml": str(yaml_path), "summary": summary}


def adopt_roboflow_yolo(roboflow_root: Path, dataset_root: Path) -> dict:
    ensure_yolo_dirs(dataset_root)
    data_yaml = roboflow_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Roboflow data.yaml missing: {data_yaml}")
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names") or FLIR_CLASSES
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    split_pairs = {
        "train": ("train", "train"),
        "valid": ("valid", "val"),
        "val": ("val", "val"),
        "test": ("test", "val"),
    }
    copied = 0
    for source_split, target_split in split_pairs.values():
        image_dir = roboflow_root / source_split / "images"
        label_dir = roboflow_root / source_split / "labels"
        if not image_dir.exists():
            image_dir = roboflow_root / "images" / source_split
            label_dir = roboflow_root / "labels" / source_split
        for image in iter_images(image_dir):
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                continue
            link_or_copy(image, dataset_root / "images" / target_split / image.name)
            link_or_copy(label, dataset_root / "labels" / target_split / label.name)
            copied += 1
    if copied == 0:
        raise RuntimeError("Roboflow FLIR fallback produced zero YOLO image/label pairs")
    config_dir = ensure_dir(get_output_dir() / "dataset_configs")
    yaml_path = write_dataset_yaml(dataset_root, config_dir / "flir_thermal.yaml", list(names))
    summary = verify_yolo_dataset(dataset_root, names=list(names), dataset_name="FLIR thermal", source="Roboflow FLIR")
    return {"root": str(dataset_root), "yaml": str(yaml_path), "summary": summary}


def prepare_flir(data_dir: Path | None = None, allow_llvip_fallback: bool = True) -> dict:
    base = Path(data_dir) if data_dir else get_data_dir()
    raw_dir = base / "flir_raw"
    dataset_root = base / "flir_thermal"
    try:
        download_flir_kaggle(raw_dir)
        result = convert_flir_coco(raw_dir, dataset_root)
        summary = result["summary"]
        status(f"FLIR dataset prepared: {summary['train_images']} train / {summary['val_images']} val images")
        return {"source": "FLIR ADAS", "thermal": result, "fallback_used": False}
    except Exception as kaggle_exc:
        LOGGER.warning("FLIR Kaggle preparation failed: %s", kaggle_exc)
        try:
            roboflow_root = download_flir_roboflow(raw_dir)
            result = adopt_roboflow_yolo(roboflow_root, dataset_root)
            summary = result["summary"]
            status(f"FLIR Roboflow fallback prepared: {summary['train_images']} train / {summary['val_images']} val images")
            return {"source": "Roboflow FLIR", "thermal": result, "fallback_used": True}
        except Exception as roboflow_exc:
            LOGGER.warning("FLIR Roboflow fallback failed: %s", roboflow_exc)
            if not allow_llvip_fallback:
                raise
            llvip = prepare_llvip(base)
            status("FLIR unavailable; using LLVIP infrared channel as thermal fallback")
            return {
                "source": "LLVIP infrared fallback",
                "thermal": llvip["infrared"],
                "pair_manifest": llvip["pair_manifest"],
                "fallback_used": True,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert FLIR ADAS thermal data to YOLO format.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-llvip-fallback", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_flir(args.data_dir, allow_llvip_fallback=not args.no_llvip_fallback), indent=2))


if __name__ == "__main__":
    main()


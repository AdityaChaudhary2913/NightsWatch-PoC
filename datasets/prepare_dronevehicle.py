import argparse
import json
import os
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import UnidentifiedImageError
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
DRONEVEHICLE_KAGGLE_SLUG = "brendanalvey/visdrone-dronevehicle"
DRONEVEHICLE_CLASSES = ["car", "truck", "bus", "van", "freight_car"]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    LOGGER.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _prepared(root: Path) -> bool:
    return bool(iter_images(root / "images" / "train")) and bool(iter_images(root / "images" / "val"))


def _extract_archives(raw_dir: Path) -> None:
    changed = True
    while changed:
        changed = False
        for archive in sorted(raw_dir.rglob("*.zip")):
            marker = archive.with_suffix(archive.suffix + ".extracted")
            if marker.exists():
                continue
            LOGGER.info("Extracting %s", archive)
            with zipfile.ZipFile(archive) as zf:
                for member in tqdm(zf.infolist(), desc=f"extract {archive.name}"):
                    zf.extract(member, archive.parent)
            marker.write_text("ok", encoding="utf-8")
            changed = True


def _download_release(raw_dir: Path) -> None:
    url = os.environ.get("DRONEVEHICLE_RELEASE_URL")
    if not url:
        raise RuntimeError("DRONEVEHICLE_RELEASE_URL is not configured; official GitHub repo exposes Baidu links")
    target = raw_dir / Path(url).name
    if target.exists():
        return
    LOGGER.info("Downloading DroneVehicle release URL: %s", url)
    urllib.request.urlretrieve(url, target)
    _extract_archives(raw_dir)


def _download_kaggle(raw_dir: Path) -> None:
    slug = os.environ.get("DRONEVEHICLE_KAGGLE_SLUG", DRONEVEHICLE_KAGGLE_SLUG)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"])
    _extract_archives(raw_dir)


def download_dronevehicle(raw_dir: Path) -> None:
    ensure_dir(raw_dir)
    if iter_images(raw_dir) and list(raw_dir.rglob("*.xml")):
        LOGGER.info("DroneVehicle raw files already present")
        return
    try:
        _download_release(raw_dir)
    except Exception as release_exc:
        LOGGER.warning("DroneVehicle GitHub release download failed: %s", release_exc)
        _download_kaggle(raw_dir)


def _split_from_path(path: Path, default: str = "train") -> str:
    text = "/".join(path.parts).lower()
    if "val" in text or "valid" in text or "validation" in text or "test" in text:
        return "val"
    if "train" in text:
        return "train"
    return default


def _modality_from_path(path: Path) -> str | None:
    text = "/".join(path.parts).lower()
    parts = {p.lower() for p in path.parts}
    if (
        "infrared" in parts
        or "thermal" in parts
        or "ir" in parts
        or "labelir" in text
        or "imgi" in text
        or "imagei" in text
        or "image_ir" in text
    ):
        return "ir"
    if (
        "rgb" in parts
        or "visible" in parts
        or "vis" in parts
        or "color" in parts
        or "labelr" in text
        or "imgr" in text
        or "imager" in text
        or "image_rgb" in text
    ):
        return "rgb"
    return None


def _normalise_class(name: str) -> int | None:
    value = name.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "van": 3,
        "freight_car": 4,
        "freight": 4,
        "feright_car": 4,
        "freightcar": 4,
    }
    return aliases.get(value)


def _index_images(raw_dir: Path) -> dict[str, dict[str, list[Path]]]:
    index: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for image in iter_images(raw_dir):
        modality = _modality_from_path(image)
        if modality is None:
            continue
        index[image.stem][modality].append(image)
    return index


def _parse_dronevehicle_xml(xml_path: Path, fallback_size: tuple[int, int]) -> list[tuple[int, tuple[float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        class_id = _normalise_class(obj.findtext("name", ""))
        if class_id is None:
            continue
        polygon = obj.find("polygon")
        if polygon is not None:
            xs, ys = [], []
            for idx in range(1, 5):
                x_text = polygon.findtext(f"x{idx}")
                y_text = polygon.findtext(f"y{idx}")
                if x_text is not None and y_text is not None:
                    xs.append(float(x_text))
                    ys.append(float(y_text))
            if xs and ys:
                boxes.append((class_id, (min(xs), min(ys), max(xs), max(ys))))
                continue
        bndbox = obj.find("bndbox")
        if bndbox is not None:
            x1 = float(bndbox.findtext("xmin", "0"))
            y1 = float(bndbox.findtext("ymin", "0"))
            x2 = float(bndbox.findtext("xmax", "0"))
            y2 = float(bndbox.findtext("ymax", "0"))
            boxes.append((class_id, (x1, y1, x2, y2)))
    return boxes


def _choose_image(candidates: list[Path], split: str) -> Path | None:
    if not candidates:
        return None
    for candidate in candidates:
        if _split_from_path(candidate) == split:
            return candidate
    return candidates[0]


def convert_dronevehicle(raw_dir: Path, output_root: Path) -> dict:
    rgb_root = output_root / "dronevehicle_rgb"
    ir_root = output_root / "dronevehicle_ir"
    manifest_path = output_root / "dronevehicle_pairs.json"
    if _prepared(rgb_root) and _prepared(ir_root):
        LOGGER.info("DroneVehicle YOLO datasets already prepared")
    else:
        for root in (rgb_root, ir_root):
            ensure_yolo_dirs(root)
        image_index = _index_images(raw_dir)
        xml_files = sorted(raw_dir.rglob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No DroneVehicle XML annotations found under {raw_dir}")
        converted = 0
        skipped_corrupt = 0
        for xml_path in tqdm(xml_files, desc="convert DroneVehicle"):
            modality = _modality_from_path(xml_path)
            if modality not in {"rgb", "ir"}:
                continue
            split = _split_from_path(xml_path)
            image_path = _choose_image(image_index.get(xml_path.stem, {}).get(modality, []), split)
            if image_path is None:
                continue
            try:
                width, height = read_image_size(image_path)
            except (UnidentifiedImageError, OSError) as exc:
                skipped_corrupt += 1
                LOGGER.warning("Skipping unreadable DroneVehicle image %s: %s", image_path, exc)
                continue
            lines = []
            for class_id, box in _parse_dronevehicle_xml(xml_path, (width, height)):
                line = yolo_line(class_id, box, width, height)
                if line:
                    lines.append(line)
            if not lines:
                continue
            root = rgb_root if modality == "rgb" else ir_root
            dst_image = root / "images" / split / image_path.name
            dst_label = root / "labels" / split / f"{image_path.stem}.txt"
            link_or_copy(image_path, dst_image)
            dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
            converted += 1
        if converted == 0:
            raise RuntimeError("DroneVehicle conversion produced zero labelled images")
        if skipped_corrupt:
            LOGGER.warning("Skipped %d unreadable DroneVehicle images during conversion", skipped_corrupt)
        pair_manifest = {"dataset": "DroneVehicle", "classes": DRONEVEHICLE_CLASSES, "train": [], "val": []}
        rgb_images = {p.stem: p for split in ("train", "val") for p in iter_images(rgb_root / "images" / split)}
        ir_images = {p.stem: p for split in ("train", "val") for p in iter_images(ir_root / "images" / split)}
        for stem in sorted(set(rgb_images) & set(ir_images)):
            rgb_image = rgb_images[stem]
            ir_image = ir_images[stem]
            split = _split_from_path(rgb_image)
            rgb_label = rgb_root / "labels" / split / f"{stem}.txt"
            ir_label = ir_root / "labels" / split / f"{stem}.txt"
            label = rgb_label if rgb_label.exists() else ir_label
            if not label.exists():
                continue
            pair_manifest[split].append(
                {
                    "visible": str(rgb_image.resolve()),
                    "infrared": str(ir_image.resolve()),
                    "label": str(label.resolve()),
                }
            )
        manifest_path.write_text(json.dumps(pair_manifest, indent=2), encoding="utf-8")
    config_dir = ensure_dir(get_output_dir() / "dataset_configs")
    rgb_yaml = write_dataset_yaml(rgb_root, config_dir / "dronevehicle_rgb.yaml", DRONEVEHICLE_CLASSES)
    ir_yaml = write_dataset_yaml(ir_root, config_dir / "dronevehicle_ir.yaml", DRONEVEHICLE_CLASSES)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({"dataset": "DroneVehicle", "classes": DRONEVEHICLE_CLASSES, "train": [], "val": []}, indent=2), encoding="utf-8")
    rgb_summary = verify_yolo_dataset(rgb_root, names=DRONEVEHICLE_CLASSES, dataset_name="DroneVehicle RGB", source="DroneVehicle")
    ir_summary = verify_yolo_dataset(ir_root, names=DRONEVEHICLE_CLASSES, dataset_name="DroneVehicle IR", source="DroneVehicle")
    return {
        "source": "DroneVehicle",
        "visible": {"root": str(rgb_root), "yaml": str(rgb_yaml), "summary": rgb_summary},
        "infrared": {"root": str(ir_root), "yaml": str(ir_yaml), "summary": ir_summary},
        "pair_manifest": str(manifest_path),
        "fallback_used": False,
    }


def prepare_dronevehicle(data_dir: Path | None = None, allow_llvip_fallback: bool = True) -> dict:
    base = Path(data_dir) if data_dir else get_data_dir()
    raw_dir = base / "dronevehicle_raw"
    try:
        download_dronevehicle(raw_dir)
        result = convert_dronevehicle(raw_dir, base)
        rgb = result["visible"]["summary"]
        ir = result["infrared"]["summary"]
        status(
            "DroneVehicle prepared: "
            f"{rgb['train_images']} RGB train / {rgb['val_images']} RGB val; "
            f"{ir['train_images']} IR train / {ir['val_images']} IR val"
        )
        return result
    except Exception as exc:
        LOGGER.warning("DroneVehicle preparation failed: %s", exc)
        if not allow_llvip_fallback:
            raise
        llvip = prepare_llvip(base)
        status("DroneVehicle unavailable; using LLVIP paired visible/infrared fallback")
        return {
            "source": "LLVIP paired fallback",
            "visible": llvip["visible"],
            "infrared": llvip["infrared"],
            "pair_manifest": llvip["pair_manifest"],
            "fallback_used": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert DroneVehicle to paired YOLO datasets.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-llvip-fallback", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare_dronevehicle(args.data_dir, allow_llvip_fallback=not args.no_llvip_fallback), indent=2))


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import os
import random
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from tqdm import tqdm

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
LLVIP_FILE_ID = "1ZM7As3u4MfcAvKWbS96RRdPFiAm9uiSG"
LLVIP_CLASSES = ["pedestrian"]


def _run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    LOGGER.info("Running command: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=True)


def _extract_archives(raw_dir: Path) -> None:
    for archive in sorted(raw_dir.glob("*")):
        if archive.suffix.lower() == ".zip":
            marker = raw_dir / f".extracted_{archive.stem}"
            if marker.exists():
                continue
            LOGGER.info("Extracting %s", archive)
            with zipfile.ZipFile(archive) as zf:
                members = zf.infolist()
                for member in tqdm(members, desc=f"extract {archive.name}"):
                    zf.extract(member, raw_dir)
            marker.write_text("ok", encoding="utf-8")


def _download_official(raw_dir: Path) -> None:
    zip_path = raw_dir / "llvip_official.zip"
    if zip_path.exists() or any(raw_dir.rglob("*.xml")):
        LOGGER.info("LLVIP raw files already present, skipping Google Drive download")
        return
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("gdown is required for the official LLVIP download") from exc
    LOGGER.info("Downloading LLVIP from Google Drive id=%s", LLVIP_FILE_ID)
    result = gdown.download(id=LLVIP_FILE_ID, output=str(zip_path), quiet=False, fuzzy=True)
    if result is None or not zip_path.exists():
        raise RuntimeError("gdown did not produce the LLVIP archive")


def _download_kaggle_fallback(raw_dir: Path) -> None:
    slug = None
    env_slug = os.environ.get("LLVIP_KAGGLE_SLUG")
    if env_slug:
        slug = env_slug
    else:
        try:
            result = _run(["kaggle", "datasets", "list", "-s", "LLVIP", "--csv"])
            rows = list(csv.DictReader(result.stdout.splitlines()))
            for row in rows:
                ref = row.get("ref") or row.get("Ref")
                if ref and "llvip" in ref.lower():
                    slug = ref
                    break
            if slug is None and rows:
                slug = rows[0].get("ref") or rows[0].get("Ref")
        except Exception as exc:
            raise RuntimeError("Kaggle LLVIP search failed") from exc
    if not slug:
        raise RuntimeError("No LLVIP Kaggle dataset slug found")
    LOGGER.info("Downloading LLVIP Kaggle fallback: %s", slug)
    _run(["kaggle", "datasets", "download", "-d", slug, "-p", str(raw_dir), "--unzip"])


def download_llvip(raw_dir: Path) -> None:
    ensure_dir(raw_dir)
    if any(raw_dir.rglob("*.xml")) and iter_images(raw_dir):
        LOGGER.info("LLVIP raw data already appears available")
        return
    try:
        _download_official(raw_dir)
        _extract_archives(raw_dir)
    except Exception as official_exc:
        LOGGER.warning("Official LLVIP download failed: %s", official_exc)
        _download_kaggle_fallback(raw_dir)
        _extract_archives(raw_dir)


def _modality_from_path(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    joined = "/".join(path.parts).lower()
    if "infrared" in parts or "ir" in parts or "lwir" in parts or "infrared" in joined:
        return "ir"
    if "visible" in parts or "vis" in parts or "rgb" in parts or "visible" in joined:
        return "vis"
    return None


def _split_from_path(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    if "val" in parts or "valid" in parts or "validation" in parts or "test" in parts:
        return "val"
    if "train" in parts or "training" in parts:
        return "train"
    return None


def _index_images(raw_dir: Path) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for image in iter_images(raw_dir):
        modality = _modality_from_path(image)
        if modality is None:
            continue
        index.setdefault(image.stem, {})[modality] = image
    return index


def _find_xml_files(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.rglob("*.xml"))


def _parse_voc_xml(xml_path: Path, fallback_size: tuple[int, int]) -> tuple[tuple[int, int], list[tuple[int, tuple[float, float, float, float]]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is not None and size.findtext("width") and size.findtext("height"):
        width = int(float(size.findtext("width", str(fallback_size[0]))))
        height = int(float(size.findtext("height", str(fallback_size[1]))))
    else:
        width, height = fallback_size
    boxes: list[tuple[int, tuple[float, float, float, float]]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "pedestrian").strip().lower()
        if name not in {"person", "pedestrian", "people", "human"}:
            continue
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x1 = float(bndbox.findtext("xmin", "0"))
        y1 = float(bndbox.findtext("ymin", "0"))
        x2 = float(bndbox.findtext("xmax", "0"))
        y2 = float(bndbox.findtext("ymax", "0"))
        boxes.append((0, (x1, y1, x2, y2)))
    return (width, height), boxes


def _record_split(xml_path: Path, rng: random.Random) -> str:
    split = _split_from_path(xml_path)
    if split:
        return split
    return "val" if rng.random() < 0.15 else "train"


def _prepared(root: Path) -> bool:
    return bool(iter_images(root / "images" / "train")) and bool(iter_images(root / "images" / "val"))


def convert_llvip(raw_dir: Path, output_root: Path) -> dict:
    vis_root = output_root / "llvip_vis"
    ir_root = output_root / "llvip_ir"
    if _prepared(vis_root) and _prepared(ir_root):
        LOGGER.info("LLVIP YOLO datasets already prepared")
    else:
        for root in (vis_root, ir_root):
            ensure_yolo_dirs(root)
        images = _index_images(raw_dir)
        xml_files = _find_xml_files(raw_dir)
        if not xml_files:
            raise FileNotFoundError(f"No LLVIP XML annotations found under {raw_dir}")
        rng = random.Random(42)
        pair_manifest = {"dataset": "LLVIP", "classes": LLVIP_CLASSES, "train": [], "val": []}
        usable = []
        for xml_path in xml_files:
            pair = images.get(xml_path.stem, {})
            if "vis" in pair and "ir" in pair:
                usable.append((xml_path, pair["vis"], pair["ir"]))
        if not usable:
            raise FileNotFoundError("Could not match LLVIP visible/infrared image pairs to XML annotations")
        for xml_path, vis_image, ir_image in tqdm(usable, desc="convert LLVIP"):
            split = _record_split(xml_path, rng)
            vis_size = read_image_size(vis_image)
            _, boxes = _parse_voc_xml(xml_path, vis_size)
            if not boxes:
                continue
            vis_label_lines = []
            ir_label_lines = []
            ir_size = read_image_size(ir_image)
            for class_id, box in boxes:
                vis_line = yolo_line(class_id, box, vis_size[0], vis_size[1])
                ir_line = yolo_line(class_id, box, ir_size[0], ir_size[1])
                if vis_line:
                    vis_label_lines.append(vis_line)
                if ir_line:
                    ir_label_lines.append(ir_line)
            if not vis_label_lines or not ir_label_lines:
                continue
            for modality, root, image_path, label_lines in (
                ("vis", vis_root, vis_image, vis_label_lines),
                ("ir", ir_root, ir_image, ir_label_lines),
            ):
                dst_image = root / "images" / split / image_path.name
                dst_label = root / "labels" / split / f"{image_path.stem}.txt"
                link_or_copy(image_path, dst_image)
                dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            pair_manifest[split].append(
                {
                    "visible": str((vis_root / "images" / split / vis_image.name).resolve()),
                    "infrared": str((ir_root / "images" / split / ir_image.name).resolve()),
                    "label": str((vis_root / "labels" / split / f"{vis_image.stem}.txt").resolve()),
                }
            )
        manifest_path = output_root / "llvip_pairs.json"
        manifest_path.write_text(json.dumps(pair_manifest, indent=2), encoding="utf-8")

    config_dir = ensure_dir(get_output_dir() / "dataset_configs")
    vis_yaml = write_dataset_yaml(vis_root, config_dir / "llvip_vis.yaml", LLVIP_CLASSES)
    ir_yaml = write_dataset_yaml(ir_root, config_dir / "llvip_ir.yaml", LLVIP_CLASSES)
    manifest_path = output_root / "llvip_pairs.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({"dataset": "LLVIP", "classes": LLVIP_CLASSES, "train": [], "val": []}, indent=2), encoding="utf-8")
    vis_summary = verify_yolo_dataset(vis_root, names=LLVIP_CLASSES, dataset_name="LLVIP visible", source="LLVIP")
    ir_summary = verify_yolo_dataset(ir_root, names=LLVIP_CLASSES, dataset_name="LLVIP infrared", source="LLVIP")
    return {
        "source": "LLVIP",
        "visible": {"root": str(vis_root), "yaml": str(vis_yaml), "summary": vis_summary},
        "infrared": {"root": str(ir_root), "yaml": str(ir_yaml), "summary": ir_summary},
        "pair_manifest": str(manifest_path),
    }


def prepare_llvip(data_dir: Path | None = None) -> dict:
    base = Path(data_dir) if data_dir else get_data_dir()
    raw_dir = base / "llvip_raw"
    output_root = base
    download_llvip(raw_dir)
    result = convert_llvip(raw_dir, output_root)
    status(
        "LLVIP prepared: "
        f"{result['visible']['summary']['train_images']} visible train / "
        f"{result['visible']['summary']['val_images']} visible val; "
        f"{result['infrared']['summary']['train_images']} IR train / "
        f"{result['infrared']['summary']['val_images']} IR val"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and convert LLVIP to YOLO format.")
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(prepare_llvip(args.data_dir), indent=2))


if __name__ == "__main__":
    main()

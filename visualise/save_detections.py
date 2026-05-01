import argparse
import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from datasets.verify_dataset import iter_images
from utils.env import ensure_dir, get_visualisation_dir
from utils.logger import get_logger, status


LOGGER = get_logger(__name__)


def _resolve_dataset_yaml(dataset_yaml: str | Path) -> tuple[Path, dict]:
    dataset_yaml = Path(dataset_yaml)
    payload = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", "."))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    return root, payload


def _validation_images(dataset_yaml: str | Path) -> list[Path]:
    root, payload = _resolve_dataset_yaml(dataset_yaml)
    val = payload.get("val", "images/val")
    val_path = Path(val)
    if not val_path.is_absolute():
        val_path = root / val_path
    return iter_images(val_path)


def save_detection_images(
    weights_path: str | Path,
    dataset_yaml: str | Path,
    model_name: str,
    output_dir: str | Path | None = None,
    num_images: int = 10,
    imgsz: int = 640,
) -> list[str]:
    from ultralytics import YOLO

    output_dir = ensure_dir(Path(output_dir) if output_dir else get_visualisation_dir() / model_name)
    images = _validation_images(dataset_yaml)[:num_images]
    if not images:
        raise FileNotFoundError(f"No validation images found for {dataset_yaml}")
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(weights_path))
    saved: list[str] = []
    for idx, image_path in enumerate(images):
        results = model.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)
        annotated = results[0].plot()
        if annotated.ndim == 3 and annotated.shape[2] == 3:
            annotated = annotated[:, :, ::-1]
        out_path = output_dir / f"{model_name}_{idx:02d}_{image_path.stem}.jpg"
        Image.fromarray(annotated).save(out_path, quality=90)
        saved.append(str(out_path))
    status(f"Detection visualisations saved: {len(saved)} images for {model_name}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Save annotated detection images for a trained model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-images", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            save_detection_images(args.weights, args.dataset_yaml, args.model_name, args.output_dir, args.num_images),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


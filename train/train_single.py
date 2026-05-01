import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from datasets.verify_dataset import ensure_yolo_dirs, write_dataset_yaml
from train.training_configs import TrainingConfig
from utils.env import ensure_dir, get_model_dir, get_output_dir
from utils.logger import get_logger, status
from utils.seed import fix_all_seeds


fix_all_seeds(42)
LOGGER = get_logger(__name__)


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if requested == "0,1" and torch.cuda.device_count() < 2:
        return "0"
    return requested


def train_with_oom_retry(model_path, dataset_yaml, epochs, output_dir, run_name, initial_batch=32):
    from ultralytics import YOLO

    batch = initial_batch
    output_dir = Path(output_dir)
    dataset_yaml = str(dataset_yaml)
    while batch >= 8:
        try:
            model = YOLO(model_path)
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=640,
                batch=batch,
                device="0,1",
                workers=4,
                seed=42,
                patience=15,
                save=True,
                project=str(output_dir),
                name=run_name,
                exist_ok=True,
                amp=True,
                cache=False,
                verbose=True,
            )
            setattr(results, "trinetra_final_batch", batch)
            setattr(results, "trinetra_device", "0,1")
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch={batch}, retrying with batch={batch//2}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch = batch // 2
            else:
                raise
    raise RuntimeError("OOM even at batch=8. Cannot proceed.")


def train_with_config(
    model_path: str,
    dataset_yaml: str | Path,
    epochs: int,
    output_dir: str | Path,
    run_name: str,
    initial_batch: int = 32,
    imgsz: int = 640,
    workers: int = 4,
    seed: int = 42,
    patience: int = 15,
    device: str = "0,1",
):
    from ultralytics import YOLO

    batch = initial_batch
    output_dir = Path(output_dir)
    dataset_yaml = str(dataset_yaml)
    training_device = _resolve_device(device)
    while batch >= 8:
        try:
            model = YOLO(model_path)
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=training_device,
                workers=workers,
                seed=seed,
                patience=patience,
                save=True,
                project=str(output_dir),
                name=run_name,
                exist_ok=True,
                amp=True,
                cache=False,
                verbose=True,
            )
            setattr(results, "trinetra_final_batch", batch)
            setattr(results, "trinetra_device", training_device)
            return results
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at batch={batch}, retrying with batch={batch//2}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                batch = batch // 2
            else:
                raise
    raise RuntimeError("OOM even at batch=8. Cannot proceed.")


def _read_results_csv(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best_row = max(rows, key=lambda row: float(row.get("metrics/mAP50(B)", row.get("metrics/mAP50", "0")) or 0))
    return {
        "best_epoch": int(float(best_row.get("epoch", len(rows) - 1))) + 1,
        "best_map50": float(best_row.get("metrics/mAP50(B)", best_row.get("metrics/mAP50", "0")) or 0),
        "best_map50_95": float(best_row.get("metrics/mAP50-95(B)", best_row.get("metrics/mAP50-95", "0")) or 0),
        "precision": float(best_row.get("metrics/precision(B)", best_row.get("metrics/precision", "0")) or 0),
        "recall": float(best_row.get("metrics/recall(B)", best_row.get("metrics/recall", "0")) or 0),
    }


def _best_weights(run_dir: Path) -> Path:
    best = run_dir / "weights" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "weights" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No trained weights found in {run_dir / 'weights'}")


def _copy_model_artifact(weights_path: Path, artifact_name: str) -> Path:
    model_dir = get_model_dir()
    target = model_dir / artifact_name
    shutil.copy2(weights_path, target)
    return target


def run_training_job(
    dataset_yaml: str | Path,
    config: TrainingConfig,
    output_dir: str | Path | None = None,
    artifact_name: str | None = None,
) -> dict:
    fix_all_seeds(config.seed)
    output_dir = Path(output_dir) if output_dir else get_output_dir() / "training_runs"
    ensure_dir(output_dir)
    LOGGER.info("Training config: %s", json.dumps(config.to_dict(), indent=2))
    results = train_with_config(
        model_path=config.model_path,
        dataset_yaml=dataset_yaml,
        epochs=config.epochs,
        output_dir=output_dir,
        run_name=config.name,
        initial_batch=config.initial_batch,
        imgsz=config.imgsz,
        workers=config.workers,
        seed=config.seed,
        patience=config.patience,
        device=config.device,
    )
    final_batch = getattr(results, "trinetra_final_batch", config.initial_batch)
    device = getattr(results, "trinetra_device", config.device)
    run_dir = output_dir / config.name
    metrics = _read_results_csv(run_dir)
    best_weights = _best_weights(run_dir)
    extended = False

    threshold = config.auto_extend_threshold_map50
    if threshold is not None and metrics.get("best_map50", 0.0) < threshold:
        extended = True
        extra_name = f"{config.name}_extra20"
        LOGGER.info(
            "Best mAP@0.5 %.3f is below %.3f; running %d more epochs from %s",
            metrics.get("best_map50", 0.0),
            threshold,
            config.extra_epochs,
            best_weights,
        )
        extra_results = train_with_config(
            model_path=str(best_weights),
            dataset_yaml=dataset_yaml,
            epochs=config.extra_epochs,
            output_dir=output_dir,
            run_name=extra_name,
            initial_batch=final_batch,
            imgsz=config.imgsz,
            workers=config.workers,
            seed=config.seed,
            patience=config.patience,
            device=config.device,
        )
        final_batch = getattr(extra_results, "trinetra_final_batch", final_batch)
        extra_run_dir = output_dir / extra_name
        extra_metrics = _read_results_csv(extra_run_dir)
        if extra_metrics.get("best_map50", 0.0) >= metrics.get("best_map50", 0.0):
            run_dir = extra_run_dir
            metrics = extra_metrics
            best_weights = _best_weights(extra_run_dir)

    artifact_name = artifact_name or f"{config.name}_best.pt"
    model_artifact = _copy_model_artifact(best_weights, artifact_name)
    summary = {
        "run_name": config.name,
        "dataset_yaml": str(dataset_yaml),
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "model_artifact": str(model_artifact),
        "final_batch": final_batch,
        "device": device,
        "extended_training": extended,
        **metrics,
    }
    status(
        "Training complete: "
        f"{config.name}, best mAP@0.5={summary.get('best_map50', 0.0):.3f} "
        f"at epoch {summary.get('best_epoch', 'unknown')}, batch={final_batch}"
    )
    return summary


def _create_synthetic_dataset(root: Path) -> Path:
    rng = np.random.default_rng(42)
    ensure_yolo_dirs(root)
    for idx in range(10):
        split = "val" if idx >= 8 else "train"
        image = (rng.random((640, 640, 3)) * 255).astype(np.uint8)
        image_path = root / "images" / split / f"synthetic_{idx:03d}.jpg"
        label_path = root / "labels" / split / f"synthetic_{idx:03d}.txt"
        Image.fromarray(image).save(image_path, quality=85)
        cx = 0.35 + 0.25 * rng.random()
        cy = 0.35 + 0.25 * rng.random()
        bw = 0.20 + 0.15 * rng.random()
        bh = 0.20 + 0.15 * rng.random()
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
    yaml_path = get_output_dir() / "dataset_configs" / "synthetic.yaml"
    return write_dataset_yaml(root, yaml_path, ["synthetic_target"])


def dry_run() -> dict:
    dataset_yaml = _create_synthetic_dataset(get_output_dir() / "synthetic_dataset")
    config = TrainingConfig(
        name="dry_run_synthetic",
        model_path="yolo11n.yaml",
        epochs=1,
        imgsz=640,
        initial_batch=8,
        workers=0,
        device="cpu",
        patience=1,
    )
    return run_training_job(dataset_yaml, config, artifact_name="dry_run_synthetic_best.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one YOLOv11 detector with OOM retry support.")
    parser.add_argument("--dataset-yaml", type=Path, default=None)
    parser.add_argument("--model-path", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--run-name", default="trinetra_train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--initial-batch", type=int, default=32)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run(), indent=2))
        return
    if args.dataset_yaml is None:
        raise ValueError("--dataset-yaml is required unless --dry-run is used")
    config = TrainingConfig(
        name=args.run_name,
        model_path=args.model_path,
        epochs=args.epochs,
        initial_batch=args.initial_batch,
        device=args.device,
    )
    print(json.dumps(run_training_job(args.dataset_yaml, config, args.output_dir), indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.env import get_log_dir
from utils.logger import get_logger, status


LOGGER = get_logger(__name__)


def _load_manifest(path: str | Path) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if "val" not in manifest:
        raise ValueError(f"Pair manifest has no val split: {path}")
    return manifest


def _image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def _load_ground_truth(label_path: str | Path, image_path: str | Path) -> list[dict]:
    width, height = _image_size(image_path)
    gts = []
    label_path = Path(label_path)
    if not label_path.exists():
        return gts
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = [float(x) for x in parts]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        gts.append({"class": int(cls), "box": [x1, y1, x2, y2]})
    return gts


def _predict(model, image_path: str | Path, imgsz: int, device) -> list[dict]:
    results = model.predict(source=str(image_path), imgsz=imgsz, device=device, verbose=False)
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    return [
        {"box": xyxy[idx].astype(float).tolist(), "score": float(conf[idx]), "class": int(cls[idx])}
        for idx in range(len(cls))
    ]


def _classwise_nms(detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if not detections:
        return []
    from torchvision.ops import nms

    kept: list[dict] = []
    by_class: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        by_class[int(det["class"])].append(det)
    for class_dets in by_class.values():
        boxes = torch.tensor([det["box"] for det in class_dets], dtype=torch.float32)
        scores = torch.tensor([det["score"] for det in class_dets], dtype=torch.float32)
        keep_idx = nms(boxes, scores, iou_threshold).cpu().tolist()
        kept.extend(class_dets[idx] for idx in keep_idx)
    return kept


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    if precision.size == 0:
        return 0.0
    ap = 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= threshold]
        ap += float(valid.max()) if valid.size else 0.0
    return ap / 101.0


def _evaluate_map(predictions: dict[str, list[dict]], ground_truths: dict[str, list[dict]], iou_thresholds: list[float]) -> float:
    classes = sorted(
        {
            int(item["class"])
            for values in list(predictions.values()) + list(ground_truths.values())
            for item in values
        }
    )
    if not classes:
        return 0.0
    aps = []
    for iou_threshold in iou_thresholds:
        for class_id in classes:
            class_preds = []
            total_gt = 0
            gt_by_image: dict[str, list[dict]] = {}
            for image_id, gts in ground_truths.items():
                class_gts = [gt for gt in gts if int(gt["class"]) == class_id]
                gt_by_image[image_id] = class_gts
                total_gt += len(class_gts)
            if total_gt == 0:
                continue
            for image_id, preds in predictions.items():
                for pred in preds:
                    if int(pred["class"]) == class_id:
                        class_preds.append((image_id, pred))
            class_preds.sort(key=lambda item: float(item[1]["score"]), reverse=True)
            matched: dict[str, set[int]] = defaultdict(set)
            tp = np.zeros(len(class_preds), dtype=np.float32)
            fp = np.zeros(len(class_preds), dtype=np.float32)
            for idx, (image_id, pred) in enumerate(class_preds):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gt_by_image.get(image_id, [])):
                    if gt_idx in matched[image_id]:
                        continue
                    value = _iou(pred["box"], gt["box"])
                    if value > best_iou:
                        best_iou = value
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    tp[idx] = 1
                    matched[image_id].add(best_gt_idx)
                else:
                    fp[idx] = 1
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / max(total_gt, 1)
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
            aps.append(_ap_from_pr(precision, recall))
    return float(np.mean(aps)) if aps else 0.0


def _metrics(predictions: dict[str, list[dict]], ground_truths: dict[str, list[dict]]) -> dict:
    thresholds = [round(v, 2) for v in np.arange(0.5, 1.0, 0.05)]
    return {
        "map50": _evaluate_map(predictions, ground_truths, [0.5]),
        "map50_95": _evaluate_map(predictions, ground_truths, thresholds),
    }


def run_late_fusion(
    eo_weights: str | Path,
    ir_weights: str | Path,
    pair_manifest: str | Path,
    output_path: str | Path | None = None,
    imgsz: int = 640,
    limit: int | None = None,
) -> dict:
    from ultralytics import YOLO

    manifest = _load_manifest(pair_manifest)
    entries = manifest["val"][:limit] if limit else manifest["val"]
    if not entries:
        raise ValueError(f"No validation pairs available in {pair_manifest}")
    device = 0 if torch.cuda.is_available() else "cpu"
    eo_model = YOLO(str(eo_weights))
    ir_model = YOLO(str(ir_weights))
    eo_preds: dict[str, list[dict]] = {}
    ir_preds: dict[str, list[dict]] = {}
    fused_preds: dict[str, list[dict]] = {}
    ground_truths: dict[str, list[dict]] = {}
    for idx, entry in enumerate(tqdm(entries, desc="late fusion")):
        image_id = str(idx)
        eo = _predict(eo_model, entry["visible"], imgsz, device)
        ir = _predict(ir_model, entry["infrared"], imgsz, device)
        fused = _classwise_nms(eo + ir, iou_threshold=0.5)
        eo_preds[image_id] = eo
        ir_preds[image_id] = ir
        fused_preds[image_id] = fused
        ground_truths[image_id] = _load_ground_truth(entry["label"], entry["visible"])
    eo_metrics = _metrics(eo_preds, ground_truths)
    ir_metrics = _metrics(ir_preds, ground_truths)
    fused_metrics = _metrics(fused_preds, ground_truths)
    best_solo_map50 = max(eo_metrics["map50"], ir_metrics["map50"])
    best_solo_map = max(eo_metrics["map50_95"], ir_metrics["map50_95"])
    table = [
        {"modality": "EO only", **eo_metrics},
        {"modality": "IR only", **ir_metrics},
        {"modality": "Late fusion", **fused_metrics},
        {
            "modality": "Δ vs best solo",
            "map50": fused_metrics["map50"] - best_solo_map50,
            "map50_95": fused_metrics["map50_95"] - best_solo_map,
        },
    ]
    payload = {
        "label": "Late-fusion NMS baseline (not reliability-aware attention fusion — that is a Phase 2 deliverable)",
        "dataset": manifest.get("dataset", "paired dataset"),
        "num_pairs": len(entries),
        "iou_threshold": 0.5,
        "table": table,
    }
    output_path = Path(output_path) if output_path else (get_log_dir() / "fusion.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    status(
        "Late-fusion baseline complete: "
        f"EO mAP@0.5={eo_metrics['map50']:.3f}, IR mAP@0.5={ir_metrics['map50']:.3f}, "
        f"fusion mAP@0.5={fused_metrics['map50']:.3f}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate late-fusion NMS baseline on paired validation frames.")
    parser.add_argument("--eo-weights", type=Path, required=True)
    parser.add_argument("--ir-weights", type=Path, required=True)
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            run_late_fusion(args.eo_weights, args.ir_weights, args.pair_manifest, args.output, limit=args.limit),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


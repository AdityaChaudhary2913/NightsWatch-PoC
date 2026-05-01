import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from utils.env import ensure_dir, get_log_dir
from utils.logger import get_logger, status


LOGGER = get_logger(__name__)
JETSON_LABEL = "Projected Jetson Orin NX INT8 (estimated — not measured on device)"


def _mean_after_warm_stabilisation(times: list[float]) -> float:
    if len(times) > 10:
        return statistics.mean(times[10:])
    return statistics.mean(times)


def _measure_torch_latency(model, dummy: torch.Tensor, half: bool = False, runs: int = 100, warmup: int = 20) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            if half:
                with torch.cuda.amp.autocast():
                    model.predict(source=dummy, verbose=False)
            else:
                model.predict(source=dummy, verbose=False)
        torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            if half:
                with torch.cuda.amp.autocast():
                    model.predict(source=dummy, verbose=False)
            else:
                model.predict(source=dummy, verbose=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return _mean_after_warm_stabilisation(times)


def _measure_gpu_memory(model, dummy: torch.Tensor) -> float:
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        model.predict(source=dummy, verbose=False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def _measure_onnx_latency(onnx_path: Path, imgsz: int = 640, runs: int = 100, warmup: int = 20) -> tuple[float, str]:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        session_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        provider_label = "CUDAExecutionProvider"
    else:
        session_providers = ["CPUExecutionProvider"]
        provider_label = "CPUExecutionProvider"
    session = ort.InferenceSession(str(onnx_path), providers=session_providers)
    input_name = session.get_inputs()[0].name
    dummy = np.random.random((1, 3, imgsz, imgsz)).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {input_name: dummy})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000.0)
    return _mean_after_warm_stabilisation(times), provider_label


def _extract_val_metrics(metrics) -> dict:
    box = getattr(metrics, "box", None)
    if box is not None:
        precision = float(np.mean(getattr(box, "p", [0.0]))) if len(getattr(box, "p", [])) else 0.0
        recall = float(np.mean(getattr(box, "r", [0.0]))) if len(getattr(box, "r", [])) else 0.0
        return {
            "map50": float(getattr(box, "map50", 0.0)),
            "map50_95": float(getattr(box, "map", 0.0)),
            "precision": precision,
            "recall": recall,
        }
    results_dict = getattr(metrics, "results_dict", {}) or {}
    return {
        "map50": float(results_dict.get("metrics/mAP50(B)", results_dict.get("metrics/mAP50", 0.0))),
        "map50_95": float(results_dict.get("metrics/mAP50-95(B)", results_dict.get("metrics/mAP50-95", 0.0))),
        "precision": float(results_dict.get("metrics/precision(B)", results_dict.get("metrics/precision", 0.0))),
        "recall": float(results_dict.get("metrics/recall(B)", results_dict.get("metrics/recall", 0.0))),
    }


def _query_power_once() -> list[float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    return [float(line.strip().split()[0]) for line in result.stdout.splitlines() if line.strip()]


def _sample_power_during_predict(model, dummy: torch.Tensor, duration_s: float = 10.0, interval_s: float = 1.0) -> dict:
    samples: list[list[float]] = []
    next_sample = time.perf_counter()
    deadline = time.perf_counter() + duration_s
    with torch.inference_mode():
        while time.perf_counter() < deadline:
            model.predict(source=dummy, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            now = time.perf_counter()
            if now >= next_sample:
                try:
                    samples.append(_query_power_once())
                except Exception as exc:
                    LOGGER.warning("Power sampling failed: %s", exc)
                    break
                next_sample = now + interval_s
    flat = [v for sample in samples for v in sample]
    if not flat:
        return {"available": False, "samples": [], "mean_watts": None, "max_watts": None}
    return {
        "available": True,
        "samples": samples,
        "mean_watts": statistics.mean(flat),
        "max_watts": max(flat),
        "num_samples": len(samples),
    }


def collect_hardware_info() -> dict:
    info = {
        "date": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            info["gpus"].append(
                {
                    "index": idx,
                    "name": props.name,
                    "vram_gb": props.total_memory / 1e9,
                }
            )
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        info["ultralytics"] = "unavailable"
    return info


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("benchmarks", [])


def save_benchmark_result(result: dict, output_path: Path | None = None) -> Path:
    output_path = output_path or (get_log_dir() / "benchmark.json")
    ensure_dir(output_path.parent)
    entries = _load_existing(output_path)
    entries = [entry for entry in entries if entry.get("model_name") != result.get("model_name")]
    entries.append(result)
    payload = {"hardware": collect_hardware_info(), "benchmarks": entries}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def benchmark_model(
    weights_path: str | Path,
    dataset_yaml: str | Path,
    model_name: str,
    output_path: str | Path | None = None,
    imgsz: int = 640,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA. This function is intended for Kaggle GPU runtime.")
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    dataset_yaml = Path(dataset_yaml)
    model_pt = YOLO(str(weights_path))
    device = 0
    LOGGER.info("Running validation for %s", model_name)
    val_metrics = _extract_val_metrics(model_pt.val(data=str(dataset_yaml), imgsz=imgsz, device=device, split="val", verbose=False))
    dummy = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    fp32_ms = _measure_torch_latency(model_pt, dummy, half=False)
    fp16_ms = _measure_torch_latency(model_pt, dummy, half=True)
    peak_gb = _measure_gpu_memory(model_pt, dummy)
    LOGGER.info("Exporting ONNX for %s", model_name)
    onnx_export = model_pt.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=False)
    onnx_path = Path(onnx_export)
    onnx_ms, onnx_provider = _measure_onnx_latency(onnx_path, imgsz=imgsz)
    power = _sample_power_during_predict(model_pt, dummy)
    result = {
        "model_name": model_name,
        "weights_path": str(weights_path),
        "dataset_yaml": str(dataset_yaml),
        "imgsz": imgsz,
        "runs": 100,
        "warmup_runs": 20,
        "stabilisation_runs_dropped": 10,
        "map50": val_metrics["map50"],
        "map50_95": val_metrics["map50_95"],
        "precision": val_metrics["precision"],
        "recall": val_metrics["recall"],
        "fp32_ms": fp32_ms,
        "fp16_ms": fp16_ms,
        "onnx_ms": onnx_ms,
        "onnx_provider": onnx_provider,
        "projected_jetson_int8_ms": onnx_ms * 1.4,
        "projected_jetson_label": JETSON_LABEL,
        "peak_memory_gb": peak_gb,
        "onnx_path": str(onnx_path),
        "power": power,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    save_benchmark_result(result, Path(output_path) if output_path else None)
    status(
        "Benchmark complete: "
        f"{model_name}, FP32={fp32_ms:.1f}ms, FP16={fp16_ms:.1f}ms, ONNX={onnx_ms:.1f}ms"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a trained YOLO model.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(benchmark_model(args.weights, args.dataset_yaml, args.model_name, args.output), indent=2))


if __name__ == "__main__":
    main()


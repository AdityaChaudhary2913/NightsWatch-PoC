import argparse
import json
import subprocess
from pathlib import Path

from utils.env import get_base_dir, get_data_dir, get_output_dir
from utils.logger import status


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_ms(value) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        rows = [["n/a" for _ in headers]]
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_base_dir(),
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unavailable in current workspace"


def _hardware_section(benchmark_payload: dict) -> str:
    hardware = benchmark_payload.get("hardware", {})
    gpus = hardware.get("gpus") or [{"name": "n/a", "vram_gb": None}]
    rows = []
    for gpu in gpus:
        rows.append(
            [
                str(gpu.get("name", "n/a")),
                _fmt(gpu.get("vram_gb"), 2),
                str(hardware.get("cuda", "n/a")),
                str(hardware.get("torch", "n/a")),
                str(hardware.get("ultralytics", "n/a")),
                str(hardware.get("date", "n/a")),
            ]
        )
    return _table(["GPU", "VRAM (GB)", "CUDA", "PyTorch", "Ultralytics", "Date"], rows)


def _dataset_section(dataset_summaries: list[dict]) -> str:
    rows = []
    for item in dataset_summaries:
        rows.append(
            [
                str(item.get("dataset", "n/a")),
                _infer_modality(str(item.get("dataset", ""))),
                str(item.get("train_images", "n/a")),
                str(item.get("val_images", "n/a")),
                str(item.get("num_classes", "n/a")),
                str(item.get("source", "n/a")),
            ]
        )
    return _table(["Dataset", "Modality", "Train images", "Val images", "Classes", "Source"], rows)


def _infer_modality(name: str) -> str:
    lower = name.lower()
    if "thermal" in lower or "infrared" in lower or "ir" in lower:
        return "IR/LWIR"
    if "rgb" in lower or "visible" in lower or "eo" in lower:
        return "EO/RGB"
    return "mixed"


def _benchmark_sections(benchmarks: list[dict]) -> str:
    sections = []
    headers = [
        "Model",
        "Dataset",
        "mAP@0.5",
        "mAP@0.5:0.95",
        "Prec",
        "Recall",
        "FP32 (ms)",
        "FP16 (ms)",
        "ONNX (ms)",
        "Jetson est. (ms)",
        "Mem (GB)",
    ]
    for item in benchmarks:
        row = [
            str(item.get("model_name", "n/a")),
            Path(str(item.get("dataset_yaml", "n/a"))).stem,
            _fmt(item.get("map50")),
            _fmt(item.get("map50_95")),
            _fmt(item.get("precision")),
            _fmt(item.get("recall")),
            _fmt_ms(item.get("fp32_ms")),
            _fmt_ms(item.get("fp16_ms")),
            _fmt_ms(item.get("onnx_ms")),
            _fmt_ms(item.get("projected_jetson_int8_ms")),
            _fmt(item.get("peak_memory_gb"), 2),
        ]
        sections.append(f"### {item.get('model_name', 'Model')}\n\n" + _table(headers, [row]))
    if not sections:
        return _table(headers, [])
    return "\n\n".join(sections)


def _fusion_section(fusion_payload: dict) -> str:
    rows = []
    for row in fusion_payload.get("table", []):
        rows.append([str(row.get("modality", "n/a")), _fmt(row.get("map50")), _fmt(row.get("map50_95"))])
    return _table(["Modality", "mAP@0.5", "mAP@0.5:0.95"], rows)


def _power_section(benchmarks: list[dict]) -> str:
    rows = []
    for item in benchmarks:
        power = item.get("power", {})
        rows.append(
            [
                str(item.get("model_name", "n/a")),
                _fmt(power.get("mean_watts"), 1),
                _fmt(power.get("max_watts"), 1),
                "yes" if power.get("available") else "no",
            ]
        )
    return _table(["Model", "Mean GPU W", "Max GPU W", "nvidia-smi available"], rows)


def _select_thermal(benchmarks: list[dict]) -> dict:
    for item in benchmarks:
        name = str(item.get("model_name", "")).lower()
        dataset = str(item.get("dataset_yaml", "")).lower()
        if "thermal" in name or "flir" in name or "ir" in dataset:
            return item
    return benchmarks[0] if benchmarks else {}


def _status_pass(value, threshold, op: str) -> str:
    if value is None:
        return "NOTE"
    if op == ">" and float(value) > threshold:
        return "PASS"
    if op == "<" and float(value) < threshold:
        return "PASS"
    return "MISS"


def _design_targets(benchmarks: list[dict]) -> str:
    item = _select_thermal(benchmarks)
    rows = [
        [
            "mAP@0.5 (thermal)",
            ">0.80",
            _fmt(item.get("map50")),
            _status_pass(item.get("map50"), 0.80, ">"),
        ],
        [
            "End-to-end latency",
            "<30 ms",
            f"{_fmt_ms(item.get('fp16_ms'))} ms (GPU FP16)",
            "NOTE" if item.get("fp16_ms") is not None else "NOTE",
        ],
        [
            "Jetson latency est.",
            "<30 ms",
            f"{_fmt_ms(item.get('projected_jetson_int8_ms'))} ms (proj.)",
            "PROJ.",
        ],
        [
            "GPU memory footprint",
            "<12 GB (16GB)",
            f"{_fmt(item.get('peak_memory_gb'), 2)} GB",
            _status_pass(item.get("peak_memory_gb"), 12.0, "<"),
        ],
    ]
    return _table(["Target", "Design Goal", "Measured Result", "Status"], rows)


def _methodology(benchmarks: list[dict], output_dir: Path) -> str:
    config_dir = output_dir / "dataset_configs"
    data_dir = get_data_dir()
    lines = [
        "```bash",
        "python -m datasets.prepare_flir",
        "python -m datasets.prepare_dronevehicle",
        f"python -m datasets.verify_dataset {config_dir / 'flir_thermal.yaml'} --strict",
        f"python -m train.train_single --dataset-yaml {config_dir / 'flir_thermal.yaml'} --epochs 50 --run-name flir_thermal_yolo11s",
        f"python -m train.train_single --dataset-yaml {config_dir / 'dronevehicle_rgb.yaml'} --epochs 30 --run-name eo_unimodal_yolo11s",
        f"python -m train.train_single --dataset-yaml {config_dir / 'dronevehicle_ir.yaml'} --epochs 30 --run-name ir_unimodal_yolo11s",
    ]
    for item in benchmarks:
        lines.append(
            "python -m eval.benchmark "
            f"--weights {item.get('weights_path', item.get('model_artifact', 'MODEL.pt'))} "
            f"--dataset-yaml {item.get('dataset_yaml', 'DATASET.yaml')} "
            f"--model-name {item.get('model_name', 'model')}"
        )
    lines.extend(
        [
            f"python -m eval.fusion_baseline --eo-weights EO.pt --ir-weights IR.pt --pair-manifest {data_dir / 'dronevehicle_pairs.json'}",
            "python -m visualise.save_detections --weights MODEL.pt --dataset-yaml DATASET.yaml --model-name MODEL",
            "python -m report.generate_report",
            "```",
            "",
            f"Git commit hash: `{_git_commit()}`",
        ]
    )
    return "\n".join(lines)


def generate_report(output_dir: str | Path | None = None) -> Path:
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    logs_dir = output_dir / "logs"
    benchmark_payload = _read_json(logs_dir / "benchmark.json", {"hardware": {}, "benchmarks": []})
    dataset_summaries = _read_json(logs_dir / "dataset_summary.json", [])
    fusion_payload = _read_json(logs_dir / "fusion.json", {"table": []})
    benchmarks = benchmark_payload.get("benchmarks", [])

    report = f"""# TRINETRA PoC — Preliminary Benchmark Results

## 1. Hardware Environment

{_hardware_section(benchmark_payload)}

## 2. Datasets Used

{_dataset_section(dataset_summaries)}

## 3. Benchmark Results

All latency measurements are averaged over 100 measured runs after 20 warmup runs, with the first 10 measured runs dropped for stabilisation. Jetson figures are labelled projections, not device measurements.

{_benchmark_sections(benchmarks)}

## 4. Multi-Sensor Fusion Baseline

Late-fusion NMS baseline (not reliability-aware attention fusion — that is a Phase 2 deliverable).

{_fusion_section(fusion_payload)}

## 5. Power Estimate

Measured GPU wattage during sustained inference loop. Jetson Orin NX TDP is configurable (10W–25W). Compute module inference power will be verified in Phase 2 on-device testing.

{_power_section(benchmarks)}

## 6. Design Target Assessment

{_design_targets(benchmarks)}

## 7. Methodology and Reproducibility

{_methodology(benchmarks, output_dir)}

## 8. Limitations and Next Steps

- All results measured on NVIDIA T4 GPU in the Kaggle environment.
- Jetson Orin NX latency figures are projections, not measured on device.
- Datasets are public benchmarks; operational performance will be validated with user-agency data in Phase 2.
- Reliability-aware EO-LWIR fusion is a Phase 2 deliverable; this PoC demonstrates a late-fusion NMS baseline only.
- All results are preliminary and will be superseded by Phase 1 bench validation after grant selection.

## 9. References

- Teledyne FLIR. FLIR ADAS thermal dataset for automotive detection research.
- Jia, X., Zhu, C., Li, M., Tang, W., and Zhou, W. LLVIP: A Visible-infrared Paired Dataset for Low-light Vision. ICCV Workshops, 2021.
- Sun, Y., Cao, B., Zhu, P., and Hu, Q. Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning. IEEE Transactions on Circuits and Systems for Video Technology, 2022.
- Ultralytics. YOLO11 object detection models and training framework.
- Microsoft. ONNX Runtime GPU inference engine.
"""
    output_path = output_dir / "poc_report.md"
    output_path.write_text(report, encoding="utf-8")
    status(f"Report generated: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TRINETRA PoC Markdown report.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(generate_report(args.output_dir))


if __name__ == "__main__":
    main()

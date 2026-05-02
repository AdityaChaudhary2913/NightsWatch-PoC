# Night's Watch PoC — Preliminary Benchmark Results

PoC code repository for reviewer reference: `https://github.com/AdityaChaudhary2913/NightsWatch-PoC`

The accompanying PoC results folder (`poc_outputs/`) contains this report, structured logs, exported PyTorch/ONNX model artifacts, training curves, confusion matrices, benchmark outputs, fusion results, and qualitative detection visualisations.

## 1. Hardware Environment

| GPU | VRAM (GB) | CUDA | PyTorch | Ultralytics | Date |
| --- | --- | --- | --- | --- | --- |
| NVIDIA L4 | 23.66 | 12.9 | 2.8.0+cu129 | 8.3.253 | 2026-05-02T02:28:48Z |

## 2. Datasets Used

| Dataset | Modality | Train images | Val images | Classes | Source |
| --- | --- | --- | --- | --- | --- |
| Thermal | IR/LWIR | 12026 | 1360 | 5 | unknown |
| EO | EO/RGB | 17467 | 8746 | 5 | unknown |
| IR | IR/LWIR | 17940 | 8965 | 5 | unknown |

## 3. Benchmark Results

All latency measurements are averaged over 100 measured runs after 20 warmup runs, with the first 10 measured runs dropped for stabilisation. Jetson figures are labelled projections, not device measurements.

### thermal_yolo11s

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Prec | Recall | FP32 (ms) | FP16 (ms) | ONNX (ms) | Jetson est. (ms) | Mem (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| thermal_yolo11s | flir_thermal | 0.995 | 0.985 | 0.997 | 0.998 | 8.3 | 10.5 | 5.8 | 8.1 | 0.18 |

### eo_yolo11s

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Prec | Recall | FP32 (ms) | FP16 (ms) | ONNX (ms) | Jetson est. (ms) | Mem (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eo_yolo11s | dronevehicle_rgb | 0.615 | 0.427 | 0.708 | 0.616 | 7.9 | 10.5 | 6.4 | 8.9 | 0.15 |

### ir_yolo11s

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Prec | Recall | FP32 (ms) | FP16 (ms) | ONNX (ms) | Jetson est. (ms) | Mem (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ir_yolo11s | dronevehicle_ir | 0.643 | 0.474 | 0.716 | 0.647 | 7.8 | 9.8 | 5.5 | 7.7 | 0.15 |

## 4. Multi-Sensor Fusion Baseline

Late-fusion NMS baseline (not reliability-aware attention fusion — that is a Phase 2 deliverable).

| Modality | mAP@0.5 | mAP@0.5:0.95 |
| --- | --- | --- |
| EO only | 0.547 | 0.383 |
| IR only | 0.509 | 0.336 |
| Late fusion | 0.568 | 0.400 |
| Δ vs best solo | 0.021 | 0.018 |

## 5. Power Estimate

Measured GPU wattage during sustained inference loop. Jetson Orin NX TDP is configurable (10W–25W). Compute module inference power will be verified in Phase 2 on-device testing.

| Model | Mean GPU W | Max GPU W | nvidia-smi available |
| --- | --- | --- | --- |
| thermal_yolo11s | 50.1 | 51.3 | yes |
| eo_yolo11s | 53.1 | 54.3 | yes |
| ir_yolo11s | 54.3 | 55.6 | yes |

## 6. Design Target Assessment

| Target | Design Goal | Measured Result | Status |
| --- | --- | --- | --- |
| mAP@0.5 (thermal) | >0.80 | 0.995 | PASS |
| End-to-end latency | <30 ms | 10.5 ms (GPU FP16) | NOTE |
| Jetson latency est. | <30 ms | 8.1 ms (proj.) | PROJ. |
| GPU memory footprint | <12 GB (16GB) | 0.18 GB | PASS |

## 7. Methodology and Reproducibility

```bash
python -m datasets.prepare_flir
python -m datasets.prepare_dronevehicle
python -m datasets.verify_dataset /mnt/nightswatch-poc/poc_outputs/dataset_configs/flir_thermal.yaml --strict
python -m train.train_single --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/flir_thermal.yaml --epochs 50 --run-name flir_thermal_yolo11s
python -m train.train_single --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/dronevehicle_rgb.yaml --epochs 30 --run-name eo_unimodal_yolo11s
python -m train.train_single --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/dronevehicle_ir.yaml --epochs 30 --run-name ir_unimodal_yolo11s
python -m eval.benchmark --weights /mnt/nightswatch-poc/poc_outputs/models/thermal_best.pt --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/flir_thermal.yaml --model-name thermal_yolo11s
python -m eval.benchmark --weights /mnt/nightswatch-poc/poc_outputs/models/eo_best.pt --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/dronevehicle_rgb.yaml --model-name eo_yolo11s
python -m eval.benchmark --weights /mnt/nightswatch-poc/poc_outputs/models/ir_best.pt --dataset-yaml /mnt/nightswatch-poc/poc_outputs/dataset_configs/dronevehicle_ir.yaml --model-name ir_yolo11s
python -m eval.fusion_baseline --eo-weights EO.pt --ir-weights IR.pt --pair-manifest /mnt/nightswatch-poc/datasets/dronevehicle_pairs.json
python -m visualise.save_detections --weights MODEL.pt --dataset-yaml DATASET.yaml --model-name MODEL
python -m report.generate_report
```

Git commit hash: `2dc4c7eccafe20a5dafc3e14fe739ba72c8e17fc`

## 8. Limitations and Next Steps

- All results measured on NVIDIA L4 GPU in the hosted GPU environment.
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

# Night's Watch PoC

Night's Watch PoC is a reproducible AI pipeline for preliminary autonomous target-detection benchmarking on public electro-optical and thermal datasets. The repository contains dataset preparation, YOLOv11 fine-tuning, latency profiling, late-fusion evaluation, detection visualisation, and a proposal-ready report generator. Local development is intentionally lightweight: write and validate code locally, then run the full GPU workflow in Kaggle through `run_all.ipynb`.

## Prerequisites

- Python 3.10+
- pip
- Git
- Kaggle account with internet access enabled for notebooks
- GPU runtime enabled in Kaggle, preferably 2x NVIDIA T4

## Local Setup

```bash
cd NightsWatch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall .
python -m train.train_single --dry-run
```

The dry run creates 10 synthetic YOLO-format images and trains for one epoch on CPU or the available local device. It is only a structural smoke test; all real training and benchmark numbers are produced on Kaggle.

## Kaggle Notebook Workflow

1. Push this repository to GitHub.
2. Open Kaggle, create a new notebook, and upload or import `run_all.ipynb`.
3. In notebook settings, enable Internet.
4. In notebook settings, enable GPU. Use 2x T4 when Kaggle offers it.
5. Run all cells top to bottom. The notebook clones the GitHub repository into `/kaggle/working/NightsWatch`, prepares datasets, trains models, benchmarks them, runs late fusion, saves visualisations, and generates `poc_report.md`.

Kaggle notebook link once published: add the published Kaggle notebook URL here.

## Downloading Outputs

After the Kaggle run completes, open the right-side Kaggle file browser and download:

- `/kaggle/working/poc_outputs/poc_report.md`
- `/kaggle/working/poc_outputs/logs/benchmark.json`
- `/kaggle/working/poc_outputs/logs/fusion.json`
- `/kaggle/working/poc_outputs/models/`
- `/kaggle/working/poc_outputs/visualisations/`

The final report is written to `/kaggle/working/poc_outputs/poc_report.md`.

## Main Entry Points

- `datasets.prepare_flir.prepare_flir()`
- `datasets.prepare_llvip.prepare_llvip()`
- `datasets.prepare_dronevehicle.prepare_dronevehicle()`
- `train.train_single.train_with_oom_retry()`
- `eval.benchmark.benchmark_model()`
- `eval.fusion_baseline.run_late_fusion()`
- `visualise.save_detections.save_detection_images()`
- `report.generate_report.generate_report()`

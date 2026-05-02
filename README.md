# Night's Watch PoC

Night's Watch PoC is a reproducible AI pipeline for preliminary autonomous target-detection benchmarking on public electro-optical and thermal datasets. The repository contains dataset preparation, YOLOv11 fine-tuning, latency profiling, late-fusion evaluation, detection visualisation, and a proposal-ready report generator. Local development is intentionally lightweight: write and validate code locally, then run the hosted workflow in Modal through split prep and training notebooks.

## Prerequisites

- Python 3.10+
- pip
- Git
- Modal notebook or comparable hosted GPU notebook
- Internet-enabled runtime with an attached persistent volume
- GPU runtime enabled in Modal, preferably `L4 x1` or better

## Local Setup

```bash
cd NightsWatch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall .
python -m train.train_single --dry-run
```

The dry run creates 10 synthetic YOLO-format images and trains for one epoch on CPU or the available local device. It is only a structural smoke test; all real training and benchmark numbers are produced in the hosted GPU notebook runtime.

## Modal Notebook Workflow

1. Push this repository to GitHub.
2. Open Modal Notebooks and attach the persistent volume at `/mnt/nightswatch-poc`.
3. For dataset preparation, use `run_prep.ipynb` on a CPU-focused profile such as `2 CPU / 4-8 GiB RAM / no GPU`.
4. For training and evaluation, use `run_train.ipynb` on the same mounted volume with a GPU profile such as `L4 x1 / 4 CPU / 16 GiB RAM`.
5. Run `run_prep.ipynb` once to prepare datasets and write `/mnt/nightswatch-poc/poc_outputs/logs/prep_manifest.json`.
6. Later, start a separate Modal notebook session with `run_train.ipynb`; it reads the prep manifest and begins training directly without rerunning dataset preparation.

`run_all.ipynb` is still available if you want a single end-to-end notebook, but the split notebooks are the recommended Modal workflow.

## Downloading Outputs

After the Modal run completes, open the file browser and download:

- `/mnt/nightswatch-poc/poc_outputs/poc_report.md`
- `/mnt/nightswatch-poc/poc_outputs/logs/benchmark.json`
- `/mnt/nightswatch-poc/poc_outputs/logs/fusion.json`
- `/mnt/nightswatch-poc/poc_outputs/models/`
- `/mnt/nightswatch-poc/poc_outputs/visualisations/`

The final report is written to `/mnt/nightswatch-poc/poc_outputs/poc_report.md`.

## Main Entry Points

- `datasets.prepare_flir.prepare_flir()`
- `datasets.prepare_llvip.prepare_llvip()`
- `datasets.prepare_dronevehicle.prepare_dronevehicle()`
- `train.train_single.train_with_oom_retry()`
- `eval.benchmark.benchmark_model()`
- `eval.fusion_baseline.run_late_fusion()`
- `visualise.save_detections.save_detection_images()`
- `report.generate_report.generate_report()`
- `run_prep.ipynb`
- `run_train.ipynb`

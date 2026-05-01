import os
from pathlib import Path


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def get_base_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/trinetra-poc")
    return Path(__file__).resolve().parent.parent


def get_output_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/poc_outputs")
    return get_base_dir() / "poc_outputs"


def get_data_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/datasets")
    return get_base_dir() / "datasets" / "raw"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_log_dir() -> Path:
    return ensure_dir(get_output_dir() / "logs")


def get_model_dir() -> Path:
    return ensure_dir(get_output_dir() / "models")


def get_visualisation_dir() -> Path:
    return ensure_dir(get_output_dir() / "visualisations")


import os
from pathlib import Path

MODAL_VOLUME_ROOT = Path(os.environ.get("NIGHTS_WATCH_VOLUME_ROOT", "/mnt/nightswatch-poc"))


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def is_modal() -> bool:
    if is_kaggle():
        return False
    return any(
        key in os.environ
        for key in ("MODAL_ENVIRONMENT", "MODAL_TASK_ID", "MODAL_IS_REMOTE")
    ) or MODAL_VOLUME_ROOT.exists()


def get_base_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/NightsWatch")
    if is_modal():
        return MODAL_VOLUME_ROOT / "NightsWatch"
    return Path(__file__).resolve().parent.parent


def get_output_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/poc_outputs")
    if is_modal():
        return MODAL_VOLUME_ROOT / "poc_outputs"
    return get_base_dir() / "poc_outputs"


def get_data_dir() -> Path:
    if is_kaggle():
        return Path("/kaggle/working/datasets")
    if is_modal():
        return MODAL_VOLUME_ROOT / "datasets"
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

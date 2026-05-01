from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    name: str
    model_path: str = "yolo11s.pt"
    epochs: int = 30
    imgsz: int = 640
    initial_batch: int = 32
    workers: int = 4
    seed: int = 42
    patience: int = 15
    device: str = "0,1"
    auto_extend_threshold_map50: float | None = None
    extra_epochs: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


FLIR_THERMAL_CONFIG = TrainingConfig(
    name="flir_thermal_yolo11s",
    epochs=50,
    auto_extend_threshold_map50=0.65,
)

EO_UNIMODAL_CONFIG = TrainingConfig(
    name="eo_unimodal_yolo11s",
    epochs=30,
)

IR_UNIMODAL_CONFIG = TrainingConfig(
    name="ir_unimodal_yolo11s",
    epochs=30,
)


def config_with_dataset(config: TrainingConfig, dataset_yaml: str | Path) -> dict:
    payload = config.to_dict()
    payload["dataset_yaml"] = str(dataset_yaml)
    return payload


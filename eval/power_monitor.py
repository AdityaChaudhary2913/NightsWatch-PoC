import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from utils.env import get_log_dir
from utils.logger import get_logger


LOGGER = get_logger(__name__)


def _query_power_once() -> list[float]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=True,
    )
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(float(line.split()[0]))
    return values


def sample_power_watts(duration_s: float = 15.0, interval_s: float = 1.0) -> dict:
    samples: list[list[float]] = []
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        try:
            samples.append(_query_power_once())
        except Exception as exc:
            LOGGER.warning("nvidia-smi power query failed: %s", exc)
            break
        time.sleep(interval_s)
    flat = [value for sample in samples for value in sample]
    if not flat:
        return {"available": False, "samples": [], "mean_watts": None, "max_watts": None}
    return {
        "available": True,
        "samples": samples,
        "mean_watts": statistics.mean(flat),
        "max_watts": max(flat),
        "num_samples": len(samples),
    }


def save_power_log(payload: dict, path: Path | None = None) -> Path:
    output_path = path or (get_log_dir() / "power.json")
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample NVIDIA GPU power draw via nvidia-smi.")
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--interval-s", type=float, default=1.0)
    args = parser.parse_args()
    payload = sample_power_watts(args.duration_s, args.interval_s)
    save_power_log(payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()


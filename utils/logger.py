import logging
import sys
from datetime import datetime


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str = "nights_watch") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def status(message: str, ok: bool = True) -> str:
    prefix = "✅ [DONE]" if ok else "❌ [FAIL]"
    line = f"{prefix} {message}"
    print(line)
    return line


def timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

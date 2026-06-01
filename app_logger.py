from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"


def _build_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return LOG_DIR / f"log_{timestamp}.txt"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = _build_log_file()
    logger = logging.getLogger("review_analyzer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.info("Logging initialized: %s", log_file)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base_name = "review_analyzer"
    return logging.getLogger(f"{base_name}.{name}" if name else base_name)

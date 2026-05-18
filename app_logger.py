from __future__ import annotations

import logging
from pathlib import Path


LOG_FILE = Path(__file__).resolve().parent / "log.txt"


def setup_logging() -> logging.Logger:
    LOG_FILE.write_text("", encoding="utf-8")
    logger = logging.getLogger("review_analyzer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.info("Logging initialized")
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base_name = "review_analyzer"
    return logging.getLogger(f"{base_name}.{name}" if name else base_name)

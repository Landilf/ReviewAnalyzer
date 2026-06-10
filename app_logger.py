from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path


LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_ENV_VAR = "REVIEW_ANALYZER_LOG_FILE"


def _build_log_file() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return LOG_DIR / f"log_{timestamp}.txt"


def _prune_old_logs(keep_last: int = 20) -> None:
    log_files = sorted(
        LOG_DIR.glob("log_*.txt"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    excess = log_files[:-keep_last] if len(log_files) > keep_last else []
    for old_log in excess:
        try:
            old_log.unlink()
        except OSError:
            continue


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    existing_log = os.getenv(LOG_ENV_VAR)
    log_file = Path(existing_log) if existing_log else _build_log_file()
    if not existing_log:
        os.environ[LOG_ENV_VAR] = str(log_file)
    logger = logging.getLogger("review_analyzer")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    existing_handler = next(
        (
            handler
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        ),
        None,
    )
    if existing_handler is None:
        logger.handlers.clear()
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.info("Logging initialized: %s", log_file)
    _prune_old_logs()
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    base_name = "review_analyzer"
    return logging.getLogger(f"{base_name}.{name}" if name else base_name)

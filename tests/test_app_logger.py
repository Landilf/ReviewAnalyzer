from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app_logger


class AppLoggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.log_dir = Path(self.temp_dir.name)
        os.environ.pop(app_logger.LOG_ENV_VAR, None)
        base_logger = app_logger.get_logger()
        base_logger.handlers.clear()

    def test_setup_logging_reuses_same_log_file_within_single_run(self) -> None:
        with patch.object(app_logger, "LOG_DIR", self.log_dir):
            logger_first = app_logger.setup_logging()
            logger_second = app_logger.setup_logging()

        log_files = sorted(self.log_dir.glob("log_*.txt"))
        self.assertEqual(len(log_files), 1)
        self.assertEqual(os.environ.get(app_logger.LOG_ENV_VAR), str(log_files[0]))
        self.assertEqual(logger_first.handlers[0].baseFilename, logger_second.handlers[0].baseFilename)

    def test_prune_old_logs_keeps_last_twenty_files(self) -> None:
        for index in range(25):
            log_path = self.log_dir / f"log_2026-06-04_10-00-{index:02d}.txt"
            log_path.write_text("test", encoding="utf-8")

        with patch.object(app_logger, "LOG_DIR", self.log_dir):
            app_logger._prune_old_logs(keep_last=20)

        remaining = sorted(self.log_dir.glob("log_*.txt"))
        self.assertEqual(len(remaining), 20)
        self.assertFalse((self.log_dir / "log_2026-06-04_10-00-00.txt").exists())


if __name__ == "__main__":
    unittest.main()

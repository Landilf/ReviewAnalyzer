from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from ui.dashboard.services import _read_dataset, analyze_frame


class ServicesTests(unittest.TestCase):
    def test_read_dataset_reads_csv_bytes(self) -> None:
        csv_bytes = "text,rating\nХорошо,5\nПлохо,2\n".encode("utf-8")

        data = _read_dataset(csv_bytes, "reviews.csv")

        self.assertEqual(list(data.columns), ["text", "rating"])
        self.assertEqual(len(data), 2)

    def test_read_dataset_raises_when_file_not_provided(self) -> None:
        with self.assertRaisesRegex(ValueError, "Загрузите CSV/XLSX-файл"):
            _read_dataset(None, None)

    @patch("ui.dashboard.services.analyze_reviews")
    def test_analyze_frame_forwards_arguments(self, analyze_reviews_mock) -> None:
        frame = pd.DataFrame({"text": ["Отлично"]})
        analyze_reviews_mock.return_value = "ok"

        result = analyze_frame(frame, method="transformer")

        self.assertEqual(result, "ok")
        analyze_reviews_mock.assert_called_once_with(
            frame,
            method="transformer",
            cancel_check=None,
        )


if __name__ == "__main__":
    unittest.main()

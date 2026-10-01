from __future__ import annotations

import unittest

import pandas as pd

from ui.dashboard.pages import _build_download_filename, _get_current_review_download


class LoaderDownloadTests(unittest.TestCase):
    def test_download_filename_includes_dataset_creation_time(self) -> None:
        filename = _build_download_filename(
            "otzovik.com",
            {"created_at": "2026-10-01_12-30-45"},
            "csv",
        )

        self.assertEqual(filename, "parsed_reviews_otzovik.com_2026-10-01_12-30-45.csv")

    def test_url_result_is_selected_for_global_download(self) -> None:
        reviews = pd.DataFrame({"text": ["Отзыв"]})

        selected, source, metadata = _get_current_review_download(
            {
                "active_input_type": "url",
                "url_reviews": reviews,
                "url_reviews_source": "otzovik.com",
                "url_reviews_meta": {"label": "url-import"},
            }
        )

        self.assertIs(selected, reviews)
        self.assertEqual(source, "otzovik.com")
        self.assertEqual(metadata["label"], "url-import")

    def test_file_and_manual_results_use_file_storage(self) -> None:
        reviews = pd.DataFrame({"text": ["Отзыв"]})
        for input_type in ("file", "manual"):
            with self.subTest(input_type=input_type):
                selected, source, _ = _get_current_review_download(
                    {"active_input_type": input_type, "file_reviews": reviews, "file_reviews_source": "reviews.csv"}
                )
                self.assertIs(selected, reviews)
                self.assertEqual(source, "reviews.csv")


if __name__ == "__main__":
    unittest.main()

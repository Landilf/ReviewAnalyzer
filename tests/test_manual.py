from __future__ import annotations

import unittest

from review_parser.manual import build_reviews_from_text


class ManualInputTests(unittest.TestCase):
    def test_build_reviews_from_text_splits_reviews_by_blank_lines(self) -> None:
        raw_text = (
            "Первый отзыв достаточно длинный, чтобы пройти фильтр.\n"
            "Здесь есть ещё немного текста.\n\n"
            "Второй отзыв тоже достаточно содержательный и не должен потеряться."
        )

        frame = build_reviews_from_text(raw_text)

        self.assertEqual(len(frame), 2)
        self.assertIn("Первый отзыв", frame.iloc[0]["text"])
        self.assertIn("Второй отзыв", frame.iloc[1]["text"])

    def test_build_reviews_from_text_filters_noise_and_short_chunks(self) -> None:
        raw_text = "Ок\n\nПодписаться на новости\n\nЭто уже полноценный отзыв с достаточной длиной."

        frame = build_reviews_from_text(raw_text)

        self.assertEqual(len(frame), 1)
        self.assertIn("полноценный отзыв", frame.iloc[0]["text"])


if __name__ == "__main__":
    unittest.main()

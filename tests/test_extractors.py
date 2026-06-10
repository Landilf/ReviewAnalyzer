from __future__ import annotations

import unittest

from review_parser.extractors import clean_text, prepare_reviews


class PrepareReviewsTests(unittest.TestCase):
    def test_prepare_reviews_deduplicates_by_url_and_generates_labels(self) -> None:
        reviews = [
            {"text": "Очень хороший отзыв", "rating": 5, "date": "2026-06-01", "url": "https://example/rev1"},
            {"text": "Очень хороший отзыв", "rating": 5, "date": "2026-06-01", "url": "https://example/rev1"},
            {"text": "Неплохой товар", "rating": 3, "date": "2026-06-02", "url": "https://example/rev2"},
        ]

        frame = prepare_reviews(reviews, source="otzovik.com", min_len=1)

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["label"].tolist(), ["positive", "neutral"])
        self.assertEqual(frame["source"].tolist(), ["otzovik.com", "otzovik.com"])

    def test_prepare_reviews_keeps_short_non_empty_reviews_when_min_len_is_one(self) -> None:
        reviews = [
            {"text": "Ок", "url": "https://example/rev1"},
            {"text": " ", "url": "https://example/rev2"},
        ]

        frame = prepare_reviews(reviews, source="otzovik.com", min_len=1)

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["text"], "Ок")

    def test_clean_text_preserves_line_breaks_for_otzovik_body(self) -> None:
        text = " Достоинства: камера\\n\\nНедостатки: нет чехла\\n Основной текст  "

        cleaned = clean_text(text, preserve_linebreaks=True)

        self.assertEqual(
            cleaned,
            "Достоинства: камера\nНедостатки: нет чехла\nОсновной текст",
        )


if __name__ == "__main__":
    unittest.main()

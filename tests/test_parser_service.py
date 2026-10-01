from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from review_parser.otzovik_crawler import PartialCrawlError
from review_parser.service import fetch_reviews_from_url


class ParserServiceTests(unittest.TestCase):
    @patch("review_parser.service.crawl_otzovik_reviews")
    def test_partial_otzovik_result_is_returned_for_download(self, crawler_mock) -> None:
        reviews = pd.DataFrame({"text": ["Первый отзыв", "Второй отзыв"]})
        crawler_mock.side_effect = PartialCrawlError(reviews, RuntimeError("соединение потеряно"))

        result = fetch_reviews_from_url("https://otzovik.com/reviews/test/", headless=False)

        self.assertEqual(len(result.reviews), 2)
        self.assertIn("Сбор прерван", result.message)
        self.assertIsNotNone(result.warning)
        assert result.warning is not None
        self.assertIn("CSV", result.warning)

    @patch("review_parser.service.crawl_otzovik_reviews")
    def test_cancelled_otzovik_result_keeps_collected_reviews(self, crawler_mock) -> None:
        reviews = pd.DataFrame({"text": ["Уже собранный отзыв"]})
        crawler_mock.side_effect = PartialCrawlError(reviews, RuntimeError("Операция отменена пользователем."))

        result = fetch_reviews_from_url("https://otzovik.com/reviews/test/", headless=False)

        self.assertEqual(result.reviews["text"].tolist(), ["Уже собранный отзыв"])
        self.assertIsNotNone(result.warning)
        assert result.warning is not None
        self.assertIn("отменена", result.warning)


if __name__ == "__main__":
    unittest.main()

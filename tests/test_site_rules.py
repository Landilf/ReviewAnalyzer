from __future__ import annotations

import unittest

from review_parser.site_rules import build_candidate_urls


class SiteRulesTests(unittest.TestCase):
    def test_build_candidate_urls_adds_otzovik_second_page_candidate(self) -> None:
        url = "https://otzovik.com/reviews/some_product/?capt4a=123"

        candidates = build_candidate_urls(url)

        self.assertIn("https://otzovik.com/reviews/some_product/", candidates)
        self.assertIn("https://otzovik.com/reviews/some_product/2/", candidates)

    def test_build_candidate_urls_adds_ozon_reviews_url(self) -> None:
        url = "https://www.ozon.ru/product/test-item-12345/?from_sku=1"

        candidates = build_candidate_urls(url)

        self.assertIn("https://www.ozon.ru/product/test-item-12345/reviews/", candidates)

    def test_build_candidate_urls_does_not_duplicate_review_detail_page(self) -> None:
        url = "https://otzovik.com/review_123456.html"

        candidates = build_candidate_urls(url)

        self.assertEqual(candidates.count(url), 1)
        self.assertEqual(len(candidates), 1)


if __name__ == "__main__":
    unittest.main()

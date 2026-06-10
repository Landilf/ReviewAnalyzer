from __future__ import annotations

import unittest
from pathlib import Path

from review_parser.extractors import extract_links, extract_reviews
from review_parser.otzovik_crawler import (
    _extract_listing_page_count,
    _extract_review_links,
    _otzovik_page_url,
    _parse_otzovik_detail_html,
)


FIXTURE_DIR = Path(__file__).resolve().parent.parent / "test_data"


def read_fixture(name: str) -> str:
    return (FIXTURE_DIR / name).read_text(encoding="utf-8")


class OtzovikFixtureTests(unittest.TestCase):
    def test_listing_fixture_contains_review_links_for_detail_pages(self) -> None:
        html = read_fixture("test_main_page.html")
        base_url = "https://otzovik.com/reviews/smartphone_xiaomi_15t/"

        links = _extract_review_links(html, base_url)

        self.assertGreaterEqual(len(links), 8)
        self.assertIn("https://otzovik.com/review_18325298.html", links)
        self.assertIn("https://otzovik.com/review_18388147.html", links)

    def test_extract_links_finds_review_links_on_listing_page(self) -> None:
        html = read_fixture("test_main_page.html")
        base_url = "https://otzovik.com/reviews/smartphone_xiaomi_15t/"

        links = extract_links(html, base_url, r"/review_\d+\.html")

        self.assertGreaterEqual(len(links), 8)
        self.assertTrue(all(link.startswith("https://otzovik.com/review_") for link in links))

    def test_detail_fixture_parses_full_review_text_rating_and_url(self) -> None:
        html = read_fixture("test_review_page.html")
        link = "https://otzovik.com/review_18325298.html"

        row = _parse_otzovik_detail_html(html, link)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["url"], link)
        self.assertEqual(row["rating"], 5.0)
        self.assertIn("Достоинства: Экран 120 Гц", row["text"])
        self.assertIn("Недостатки: Очень скользкий корпус", row["text"])
        self.assertIn("в восторге", row["text"])
        self.assertNotIn("Достоинства: Достоинства:", row["text"])
        self.assertNotIn("Недостатки: Недостатки:", row["text"])

    def test_analog_detail_fixture_parses_without_label_duplication(self) -> None:
        html = read_fixture("test_main_page_analog.html")
        link = "https://otzovik.com/review_18388147.html"

        row = _parse_otzovik_detail_html(html, link)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["rating"], 4.0)
        self.assertIn("Достоинства: батарея долго держит", row["text"])
        self.assertIn("Недостатки: нет разъема для наушников", row["text"])
        self.assertNotIn("Достоинства: Достоинства:", row["text"])
        self.assertNotIn("Недостатки: Недостатки:", row["text"])

    def test_extract_reviews_can_build_frame_from_detail_fixture(self) -> None:
        html = read_fixture("test_review_page.html")

        frame = extract_reviews(html, "otzovik.com", limit=1)

        self.assertEqual(len(frame), 1)
        self.assertIn("text", frame.columns)
        self.assertIn("rating", frame.columns)
        self.assertEqual(frame.iloc[0]["label"], "positive")

    def test_listing_page_count_and_page_url_helpers_behave_consistently(self) -> None:
        html = read_fixture("test_main_page.html")
        listing_url = "https://otzovik.com/reviews/smartphone_xiaomi_15t/"

        page_count = _extract_listing_page_count(html, listing_url)
        page_2_url = _otzovik_page_url(listing_url, 2)

        self.assertEqual(page_count, 1)
        self.assertEqual(page_2_url, "https://otzovik.com/reviews/smartphone_xiaomi_15t/2/")


if __name__ == "__main__":
    unittest.main()

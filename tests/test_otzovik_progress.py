from __future__ import annotations

import asyncio
import unittest

from review_parser.otzovik_crawler import _format_duration, _publish_checkpoint, _wait_for_captcha_solve


class OtzovikProgressTests(unittest.TestCase):
    def test_format_duration_uses_readable_russian_units(self) -> None:
        self.assertEqual(_format_duration(0), "0 сек")
        self.assertEqual(_format_duration(65), "1 мин, 5 сек")
        self.assertEqual(_format_duration(3661), "1 ч, 1 мин, 1 сек")

    def test_captcha_wait_honors_cancellation_without_waiting(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "отменена"):
            asyncio.run(_wait_for_captcha_solve(None, lambda *_: None, cancel_check=lambda: True))

    def test_checkpoint_contains_reviews_collected_so_far(self) -> None:
        checkpoints = []

        _publish_checkpoint(
            [{"text": "Хороший товар", "rating": 5, "url": "https://otzovik.com/review_1.html"}],
            None,
            checkpoints.append,
        )

        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["text"].tolist(), ["Хороший товар"])


if __name__ == "__main__":
    unittest.main()

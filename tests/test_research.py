from __future__ import annotations

import unittest

import pandas as pd

from ui.dashboard.research import _compare_periods


class ResearchHelpersTests(unittest.TestCase):
    def test_compare_periods_calculates_review_count_shares_and_confidence(self) -> None:
        frame = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-06-01"), "predicted_sentiment": "negative", "confidence": 0.6},
                {"date": pd.Timestamp("2026-06-02"), "predicted_sentiment": "positive", "confidence": 0.9},
                {"date": pd.Timestamp("2026-06-10"), "predicted_sentiment": "negative", "confidence": 0.8},
            ]
        )

        periods = _compare_periods(
            frame,
            (pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-03")),
            (pd.Timestamp("2026-06-09"), pd.Timestamp("2026-06-11")),
        )

        self.assertEqual(periods.loc[0, "reviews"], 2)
        self.assertAlmostEqual(periods.loc[0, "negative_share"], 0.5)
        self.assertAlmostEqual(periods.loc[0, "positive_share"], 0.5)
        self.assertAlmostEqual(periods.loc[1, "avg_confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()

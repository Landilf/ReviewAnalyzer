from __future__ import annotations

import unittest

import pandas as pd

from analysis_helpers.insights import build_html_report, build_insights, build_recommendations


class InsightsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews = pd.DataFrame(
            [
                {
                    "review_id": 1,
                    "text": "Камера хорошая, но батарея слабая",
                    "predicted_sentiment": "negative",
                    "confidence": 0.52,
                    "category": "Смартфоны",
                },
                {
                    "review_id": 2,
                    "text": "Экран отличный",
                    "predicted_sentiment": "positive",
                    "confidence": 0.93,
                    "category": "Смартфоны",
                },
                {
                    "review_id": 3,
                    "text": "Батарея держит плохо",
                    "predicted_sentiment": "negative",
                    "confidence": 0.84,
                    "category": "Смартфоны",
                },
            ]
        )
        self.aspect_stats = pd.DataFrame(
            [
                {"aspect": "батарея", "mentions": 3, "negative": 2, "neutral": 0, "positive": 1, "negative_share": 0.667},
                {"aspect": "экран", "mentions": 2, "negative": 0, "neutral": 0, "positive": 2, "negative_share": 0.0},
            ]
        )

    def test_build_insights_returns_dominant_sentiment_problem_aspect_and_confidence_hint(self) -> None:
        insights = build_insights(self.reviews, self.aspect_stats)

        self.assertTrue(any("Преобладающая тональность" in item for item in insights))
        self.assertTrue(any("Наиболее проблемный аспект" in item for item in insights))
        self.assertTrue(any("низкой уверенностью" in item for item in insights))

    def test_build_recommendations_returns_top_problem_aspects(self) -> None:
        recommendations = build_recommendations(self.aspect_stats)

        self.assertGreaterEqual(len(recommendations), 1)
        self.assertIn("батарея", recommendations[0])

    def test_build_html_report_contains_model_metrics_and_json_payload(self) -> None:
        topics = pd.DataFrame([{"topic": "Тема 1", "review_count": 2, "share": 0.67}])
        html_report = build_html_report(
            self.reviews,
            self.aspect_stats,
            topics,
            insights=["Вывод 1"],
            recommendations=["Рекомендация 1"],
            model_name="RuBERT tiny",
            model_metrics={"accuracy": 0.91},
            confusion=pd.DataFrame([[2, 1]], index=["negative"], columns=["negative", "positive"]),
            source_name="otzovik.com",
            filters={"sentiments": ["negative", "positive"]},
            options={},
        )

        self.assertIn("Экспериментальный HTML-отчёт ReviewAnalyzer", html_report)
        self.assertIn("RuBERT tiny", html_report)
        self.assertIn("accuracy", html_report)
        self.assertIn("otzovik.com", html_report)
        self.assertIn("Рекомендация 1", html_report)
        self.assertIn("sentiment_distribution", html_report)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from analysis_helpers.pipeline import analyze_reviews, ensure_review_columns


class EnsureReviewColumnsTests(unittest.TestCase):
    def test_renames_first_textual_column_and_generates_service_fields(self) -> None:
        source = pd.DataFrame(
            {
                "review_body": ["Отличный товар", " ", None, "Батарея держит долго"],
                "label": ["Позитивный", "Негативный", "Нейтральный", "label_0"],
            }
        )

        prepared = ensure_review_columns(source)

        self.assertIn("text", prepared.columns)
        self.assertEqual(prepared["text"].tolist(), ["Отличный товар", "Батарея держит долго"])
        self.assertEqual(prepared["label"].tolist(), ["positive", "negative"])
        self.assertEqual(prepared["category"].tolist(), ["Общая категория", "Общая категория"])
        self.assertEqual(prepared["rating"].tolist(), [5, 2])
        self.assertEqual(prepared["review_id"].tolist(), [1, 2])
        self.assertEqual(prepared["word_count"].tolist(), [2, 3])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prepared["date"]))

    def test_raises_when_no_textual_column_exists(self) -> None:
        source = pd.DataFrame({"rating": [5, 4], "score": [0.5, 0.9]})

        with self.assertRaisesRegex(ValueError, "нужен столбец с текстом"):
            ensure_review_columns(source)


class AnalyzeReviewsTests(unittest.TestCase):
    @patch("analysis_helpers.pipeline.sentiment_with_transformer")
    @patch("analysis_helpers.pipeline._extract_aspects")
    @patch("analysis_helpers.pipeline.build_topic_frames")
    @patch("analysis_helpers.pipeline.build_clusters")
    @patch("analysis_helpers.pipeline.build_confusion_matrix")
    @patch("analysis_helpers.pipeline.build_model_comparison")
    def test_analyze_reviews_builds_expected_columns(
        self,
        build_model_comparison_mock,
        build_confusion_matrix_mock,
        build_clusters_mock,
        build_topic_frames_mock,
        extract_aspects_mock,
        sentiment_with_transformer_mock,
    ) -> None:
        dataset = pd.DataFrame(
            {
                "text": ["Очень хороший экран", "Звук посредственный"],
                "label": ["positive", "negative"],
                "rating": [5, 2],
            }
        )

        sentiment_with_transformer_mock.return_value = lambda batch, **_: [
            {"label": "LABEL_2", "score": 0.91},
            {"label": "LABEL_0", "score": 0.87},
        ][: len(batch)]
        extract_aspects_mock.return_value = [["экран"], ["звук"]]
        build_topic_frames_mock.return_value = (
            pd.DataFrame([{"topic": "Тема 1", "review_count": 2, "share": 1.0}]),
            pd.DataFrame([{"topic": "Тема 1", "term": "экран", "weight": 1.0}]),
            ["Тема 1", "Тема 1"],
        )
        build_clusters_mock.return_value = ["Кластер 1", "Кластер 2"]
        build_confusion_matrix_mock.return_value = pd.DataFrame([[1, 0], [0, 1]])
        build_model_comparison_mock.return_value = pd.DataFrame(
            [{"model": "RuBERT tiny", "metric": "accuracy", "value": 1.0}]
        )

        result = analyze_reviews(dataset, method="transformer")

        self.assertEqual(result.model_name, "RuBERT tiny")
        self.assertEqual(result.reviews["predicted_sentiment"].tolist(), ["positive", "negative"])
        self.assertEqual(result.reviews["confidence"].tolist(), [0.91, 0.87])
        self.assertEqual(result.reviews["aspects_text"].tolist(), ["экран", "звук"])
        self.assertEqual(result.reviews["topic"].tolist(), ["Тема 1", "Тема 1"])
        self.assertEqual(result.reviews["cluster"].tolist(), ["Кластер 1", "Кластер 2"])
        self.assertIn("accuracy", result.metrics)

    def test_cancel_check_interrupts_analysis_before_processing(self) -> None:
        dataset = pd.DataFrame({"text": ["Отзыв"], "label": ["positive"]})

        with self.assertRaisesRegex(RuntimeError, "Операция отменена"):
            analyze_reviews(dataset, cancel_check=lambda: True)


if __name__ == "__main__":
    unittest.main()

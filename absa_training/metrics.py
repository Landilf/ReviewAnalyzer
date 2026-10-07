"""Metrics for the taxonomy and aspect-sentiment components of ABSA."""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import accuracy_score, classification_report

from absa_training.data import SENTIMENTS


def _report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )


def build_absa_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    taxonomies: list[str],
) -> dict:
    """Return joint, taxonomy, and taxonomy-conditioned sentiment metrics."""
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Количество истинных и предсказанных меток должно совпадать.")
    true_taxonomy = [label.split("::", maxsplit=1)[0] for label in true_labels]
    predicted_taxonomy = [label.split("::", maxsplit=1)[0] for label in predicted_labels]
    true_sentiment = [label.split("::", maxsplit=1)[1] for label in true_labels]
    predicted_sentiment = [label.split("::", maxsplit=1)[1] for label in predicted_labels]
    joint_labels = [f"{taxonomy}::{sentiment}" for taxonomy in taxonomies for sentiment in sorted(SENTIMENTS)]
    per_taxonomy: dict[str, dict] = {}
    for taxonomy in taxonomies:
        indexes = [index for index, value in enumerate(true_taxonomy) if value == taxonomy]
        if not indexes:
            continue
        category_true = [true_sentiment[index] for index in indexes]
        category_predicted = [predicted_sentiment[index] for index in indexes]
        per_taxonomy[taxonomy] = {
            "support": len(indexes),
            "joint_accuracy": accuracy_score(
                [true_labels[index] for index in indexes],
                [predicted_labels[index] for index in indexes],
            ),
            "sentiment": _report(category_true, category_predicted, sorted(SENTIMENTS)),
        }
    return {
        "support": len(true_labels),
        "joint": _report(true_labels, predicted_labels, joint_labels),
        "taxonomy": _report(true_taxonomy, predicted_taxonomy, taxonomies),
        "sentiment": _report(true_sentiment, predicted_sentiment, sorted(SENTIMENTS)),
        "per_taxonomy": per_taxonomy,
        "true_label_distribution": dict(Counter(true_labels)),
    }


def primary_scores(metrics: dict) -> dict[str, float]:
    """Compact epoch scores used in history.csv and checkpoint selection."""
    return {
        "joint_macro_f1": float(metrics["joint"]["macro avg"]["f1-score"]),
        "joint_weighted_f1": float(metrics["joint"]["weighted avg"]["f1-score"]),
        "taxonomy_macro_f1": float(metrics["taxonomy"]["macro avg"]["f1-score"]),
        "sentiment_macro_f1": float(metrics["sentiment"]["macro avg"]["f1-score"]),
    }

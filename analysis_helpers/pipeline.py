from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis_helpers.aspects import extract_aspects_simple
from analysis_helpers.config import RANDOM_STATE
from analysis_helpers.evaluation import evaluate
from analysis_helpers.sentiment import sentiment_with_transformer, sentiment_without_spacy
from analysis_helpers.topics import describe_topics, topic_modeling


CancelCheck = Callable[[], bool] | None


SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_LABELS_RU = {
    "negative": "Негативные",
    "neutral": "Нейтральные",
    "positive": "Позитивные",
}


@dataclass
class AnalysisResult:
    reviews: pd.DataFrame
    metrics: dict[str, float]
    topics: pd.DataFrame
    topic_terms: pd.DataFrame
    aspect_stats: pd.DataFrame
    model_comparison: pd.DataFrame
    confusion: pd.DataFrame
    model_name: str


def normalize_sentiment(label: str) -> str:
    normalized = str(label).strip().lower()
    mapping = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "негативный": "negative",
        "нейтральный": "neutral",
        "позитивный": "positive",
        "негатив": "negative",
        "нейтрально": "neutral",
        "позитив": "positive",
    }
    return mapping.get(normalized, normalized)


def ensure_review_columns(data: pd.DataFrame) -> pd.DataFrame:
    prepared = data.copy()
    if "text" not in prepared.columns:
        text_candidates = [column for column in prepared.columns if prepared[column].dtype == "object"]
        if not text_candidates:
            raise ValueError("В наборе данных нужен столбец с текстом отзыва.")
        prepared = prepared.rename(columns={text_candidates[0]: "text"})

    prepared["text"] = prepared["text"].fillna("").astype(str)
    prepared = prepared[prepared["text"].str.strip().ne("")].reset_index(drop=True)

    if "label" in prepared.columns:
        prepared["label"] = prepared["label"].map(normalize_sentiment)
    if "category" not in prepared.columns:
        prepared["category"] = "Общая категория"
    if "rating" not in prepared.columns:
        prepared["rating"] = prepared.get("label", pd.Series(["neutral"] * len(prepared))).map(
            {"negative": 2, "neutral": 3, "positive": 5}
        )
    if "date" not in prepared.columns:
        prepared["date"] = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(prepared), freq="D")
    else:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    prepared["review_id"] = range(1, len(prepared) + 1)
    prepared["text_length"] = prepared["text"].str.len()
    prepared["word_count"] = prepared["text"].str.split().str.len()
    return prepared


def analyze_reviews(
    data: pd.DataFrame,
    method: str = "transformer",
    cancel_check: CancelCheck = None,
) -> AnalysisResult:
    _check_cancel(cancel_check)
    reviews = ensure_review_columns(data)
    texts = reviews["text"].tolist()
    labels = reviews["label"].tolist() if "label" in reviews.columns else None

    if method == "classic" and not _can_train_classic_model(labels):
        method = "transformer"

    if method == "transformer":
        classifier = sentiment_with_transformer()
        predictions = []
        confidences = []
        for start in range(0, len(texts), 16):
            _check_cancel(cancel_check)
            batch = texts[start:start + 16]
            raw_predictions = classifier(batch, truncation=True, max_length=512)
            predictions.extend(normalize_sentiment(p["label"]) for p in raw_predictions)
            confidences.extend(round(float(p["score"]), 4) for p in raw_predictions)
        model_name = "RuBERT tiny"
    else:
        _check_cancel(cancel_check)
        predictions, confidences, metrics = _predict_with_classic_model(texts, labels, cancel_check)
        model_name = "TF-IDF + Logistic Regression"

    _check_cancel(cancel_check)
    reviews["aspects"] = _extract_aspects(texts)

    reviews["predicted_sentiment"] = predictions
    reviews["confidence"] = confidences

    if labels is not None and method == "transformer":
        metrics = evaluate(labels, reviews["predicted_sentiment"].tolist())
    elif labels is None:
        metrics = {}

    reviews["aspects_text"] = reviews["aspects"].apply(lambda values: ", ".join(values))

    _check_cancel(cancel_check)
    aspect_stats = build_aspect_stats(reviews)
    topics, topic_terms, topic_labels = build_topic_frames(reviews, cancel_check=cancel_check)
    reviews["topic"] = topic_labels
    reviews["cluster"] = build_clusters(reviews, cancel_check=cancel_check)

    confusion = build_confusion_matrix(reviews)
    model_comparison = build_model_comparison(reviews, metrics, model_name)

    return AnalysisResult(
        reviews=reviews,
        metrics=metrics,
        topics=topics,
        topic_terms=topic_terms,
        aspect_stats=aspect_stats,
        model_comparison=model_comparison,
        confusion=confusion,
        model_name=model_name,
    )


def build_aspect_stats(reviews: pd.DataFrame, limit: int = 40) -> pd.DataFrame:
    rows = []
    for _, review in reviews.iterrows():
        for aspect in review["aspects"]:
            rows.append(
                {
                    "aspect": aspect,
                    "sentiment": review["predicted_sentiment"],
                    "review_id": review["review_id"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["aspect", "mentions", "negative", "neutral", "positive", "negative_share"])

    aspect_mentions = pd.DataFrame(rows)
    pivot = pd.crosstab(aspect_mentions["aspect"], aspect_mentions["sentiment"])
    for sentiment in SENTIMENT_ORDER:
        if sentiment not in pivot.columns:
            pivot[sentiment] = 0

    pivot["mentions"] = pivot[SENTIMENT_ORDER].sum(axis=1)
    pivot["negative_share"] = (pivot["negative"] / pivot["mentions"]).round(3)
    return (
        pivot.reset_index()
        .sort_values(["mentions", "negative_share"], ascending=[False, False])
        .head(limit)
    )


def build_topic_frames(
    reviews: pd.DataFrame,
    n_topics: int = 5,
    cancel_check: CancelCheck = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    _check_cancel(cancel_check)
    topic_input = reviews["aspects"].apply(lambda values: " ".join(values) if values else "").tolist()
    if sum(bool(text.strip()) for text in topic_input) < 3:
        topic_input = reviews["text"].tolist()

    try:
        _check_cancel(cancel_check)
        lda, vectorizer, doc_topics = topic_modeling(topic_input, n_topics=n_topics)
    except ValueError:
        return (
            pd.DataFrame(columns=["topic", "review_count", "share"]),
            pd.DataFrame(columns=["topic", "term", "weight"]),
            ["Тема не определена"] * len(reviews),
        )

    dominant_topics = doc_topics.argmax(axis=1)
    topic_labels = [f"Тема {topic_idx + 1}" for topic_idx in dominant_topics]
    counts = Counter(dominant_topics)
    topics = pd.DataFrame(
        [
            {
                "topic": f"Тема {topic_idx + 1}",
                "review_count": count,
                "share": round(count / len(reviews), 3),
            }
            for topic_idx, count in sorted(counts.items())
        ]
    )
    topic_terms = describe_topics(lda, vectorizer, top_words=8)
    return topics, topic_terms, topic_labels


def build_confusion_matrix(reviews: pd.DataFrame) -> pd.DataFrame:
    if "label" not in reviews.columns:
        return pd.DataFrame()
    return pd.crosstab(
        reviews["label"],
        reviews["predicted_sentiment"],
        rownames=["Истинная метка"],
        colnames=["Предсказание"],
        dropna=False,
    )


def build_model_comparison(reviews: pd.DataFrame, metrics: dict[str, float], model_name: str) -> pd.DataFrame:
    if not metrics:
        return pd.DataFrame(columns=["model", "metric", "value"])
    return pd.DataFrame(
        [{"model": model_name, "metric": metric, "value": value} for metric, value in metrics.items()]
    )


def build_clusters(
    reviews: pd.DataFrame,
    max_clusters: int = 5,
    cancel_check: CancelCheck = None,
) -> list[str]:
    _check_cancel(cancel_check)
    if len(reviews) < 3:
        return ["Кластер 1"] * len(reviews)

    n_clusters = min(max_clusters, len(reviews))
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    try:
        _check_cancel(cancel_check)
        matrix = vectorizer.fit_transform(reviews["text"])
        _check_cancel(cancel_check)
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(matrix)
    except ValueError:
        return ["Кластер не определён"] * len(reviews)
    return [f"Кластер {label + 1}" for label in labels]

def _predict_with_classic_model(
    texts: list[str],
    labels: Iterable[str] | None,
    cancel_check: CancelCheck = None,
) -> tuple[list[str], list[float], dict[str, float]]:
    _check_cancel(cancel_check)
    vectorizer, model = sentiment_without_spacy()
    if labels is None:
        raise ValueError("Классический режим требует столбец label для обучения модели.")

    labels = list(labels)
    stratify = labels if min(Counter(labels).values()) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    x_train_vec = vectorizer.fit_transform(x_train)
    _check_cancel(cancel_check)
    model.fit(x_train_vec, y_train)

    x_test_vec = vectorizer.transform(x_test)
    test_predictions = model.predict(x_test_vec)
    metrics = evaluate(y_test, test_predictions)

    x_all_vec = vectorizer.transform(texts)
    predictions = model.predict(x_all_vec)
    probabilities = model.predict_proba(x_all_vec)
    confidences = [round(float(max(row)), 4) for row in probabilities]
    return list(predictions), confidences, metrics


def _can_train_classic_model(labels: Iterable[str] | None) -> bool:
    if labels is None:
        return False
    labels = [label for label in labels if pd.notna(label)]
    if len(labels) < 4:
        return False
    counts = Counter(labels)
    return len(counts) >= 2 and min(counts.values()) >= 2

def _extract_aspects(texts: list[str]) -> list[list[str]]:
    return extract_aspects_simple(texts)


def _check_cancel(cancel_check: CancelCheck) -> None:
    if cancel_check and cancel_check():
        raise RuntimeError("Операция отменена пользователем.")

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from app_logger import get_logger
from analysis_helpers.config import RANDOM_STATE
from analysis_helpers.pipeline import normalize_sentiment


logger = get_logger("ui.developer")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "dns_electro_sentiment.joblib"


def render_developer_panel(current_reviews: pd.DataFrame | None) -> None:
    st.subheader("Для разработчика")
    st.markdown(
        "Здесь можно дообучить классическую модель под конкретную категорию (например, DNS + электроника), "
        "сохранить её и применить к текущему набору отзывов."
    )
    _render_training_block()
    st.divider()
    _render_inference_block(current_reviews)


def _render_training_block() -> None:
    st.markdown("#### Дообучение модели")
    with st.container(border=True):
        dataset = st.file_uploader(
            "Датасет для дообучения (CSV или Excel, с `text` и `label`)",
            type=["csv", "xlsx", "xls"],
            key="dev_train_dataset",
        )
        use_category_filter = st.checkbox("Фильтровать по категории перед обучением", value=True)
        category_field = st.text_input("Название столбца категории", value="category")
        category_value = st.text_input("Значение категории", value="Электроника")
        keyword_filter = st.text_input("Ключевые слова в тексте (через запятую)", value="смартфон, ноутбук, наушники, телевизор")
        train_clicked = st.button("Дообучить и сохранить модель", type="primary")

    if not train_clicked:
        return
    if dataset is None:
        st.error("Загрузите датасет для обучения.")
        return

    try:
        frame = _read_dataset(dataset)
        logger.info("Developer training started: rows=%s columns=%s", len(frame), list(frame.columns))
        prepared = _prepare_training_frame(
            frame,
            use_category_filter=use_category_filter,
            category_field=category_field,
            category_value=category_value,
            keyword_filter=keyword_filter,
        )
        metrics, artifact = _train_model(prepared)
        _save_artifact(artifact)
        logger.info("Developer training finished: rows=%s metrics=%s", len(prepared), metrics)
    except Exception as exc:
        logger.exception("Developer training failed: %s", exc)
        st.error(f"Не удалось дообучить модель: {exc}")
        return

    st.success(f"Модель сохранена в `{MODEL_PATH}`")
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)


def _render_inference_block(current_reviews: pd.DataFrame | None) -> None:
    st.markdown("#### Применение дообученной модели")
    with st.container(border=True):
        st.write(f"Файл модели: `{MODEL_PATH}`")
        apply_clicked = st.button("Применить к текущим отзывам")

    if not apply_clicked:
        return
    if current_reviews is None or current_reviews.empty:
        st.error("Сейчас нет загруженных отзывов для применения модели.")
        return
    if not MODEL_PATH.exists():
        st.error("Сначала дообучите модель и сохраните её.")
        return

    try:
        artifact = _load_artifact()
        result = _apply_model(current_reviews, artifact)
    except Exception as exc:
        logger.exception("Developer inference failed: %s", exc)
        st.error(f"Не удалось применить модель: {exc}")
        return

    st.success(f"Обработано {len(result)} отзывов дообученной моделью.")
    st.dataframe(
        result[["text", "dev_predicted_sentiment", "dev_confidence"]].head(80),
        use_container_width=True,
        hide_index=True,
    )
    distribution = result["dev_predicted_sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
    st.dataframe(distribution, use_container_width=True, hide_index=True)


def _read_dataset(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def _prepare_training_frame(
    frame: pd.DataFrame,
    use_category_filter: bool,
    category_field: str,
    category_value: str,
    keyword_filter: str,
) -> pd.DataFrame:
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError("В датасете должны быть столбцы `text` и `label`.")

    prepared = frame.copy()
    prepared["text"] = prepared["text"].fillna("").astype(str)
    prepared["label"] = prepared["label"].map(normalize_sentiment)
    prepared = prepared[prepared["text"].str.strip().ne("")]
    prepared = prepared[prepared["label"].isin(["negative", "neutral", "positive"])]

    if use_category_filter and category_field in prepared.columns and category_value.strip():
        prepared = prepared[prepared[category_field].astype(str).str.contains(category_value, case=False, na=False)]

    tokens = [token.strip().lower() for token in keyword_filter.split(",") if token.strip()]
    if tokens:
        mask = prepared["text"].str.lower().apply(lambda text: any(token in text for token in tokens))
        prepared = prepared[mask]

    if len(prepared) < 30:
        raise ValueError("После фильтрации осталось слишком мало отзывов. Нужно хотя бы 30.")

    label_counts = prepared["label"].value_counts()
    if len(label_counts) < 2 or label_counts.min() < 3:
        raise ValueError("Недостаточно баланса классов после фильтрации (минимум 3 отзыва на класс).")
    return prepared.reset_index(drop=True)


def _train_model(prepared: pd.DataFrame):
    x_train, x_test, y_train, y_test = train_test_split(
        prepared["text"].tolist(),
        prepared["label"].tolist(),
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=prepared["label"].tolist(),
    )
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    model = LogisticRegression(max_iter=2500, class_weight="balanced")

    x_train_vec = vectorizer.fit_transform(x_train)
    model.fit(x_train_vec, y_train)
    x_test_vec = vectorizer.transform(x_test)
    predictions = model.predict(x_test_vec)

    metrics = {
        "rows_used": len(prepared),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(precision_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, predictions, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(y_test, predictions, average="weighted", zero_division=0), 4),
    }
    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "trained_at": datetime.utcnow().isoformat(),
        "rows_used": len(prepared),
    }
    return metrics, artifact


def _save_artifact(artifact: dict) -> None:
    from joblib import dump

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(artifact, MODEL_PATH)


def _load_artifact() -> dict:
    from joblib import load

    return load(MODEL_PATH)


def _apply_model(reviews: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    frame = reviews.copy()
    vectorizer = artifact["vectorizer"]
    model = artifact["model"]

    vectors = vectorizer.transform(frame["text"].fillna("").astype(str))
    preds = model.predict(vectors)
    proba = model.predict_proba(vectors)
    confidences = [round(float(max(row)), 4) for row in proba]

    frame["dev_predicted_sentiment"] = preds
    frame["dev_confidence"] = confidences
    return frame

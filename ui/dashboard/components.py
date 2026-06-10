from __future__ import annotations

import pandas as pd
import streamlit as st

from app_control import request_cancel


DISPLAY_COLUMN_NAMES = {
    "review_id": "ID отзыва",
    "text": "Текст отзыва",
    "label": "Истинная метка",
    "predicted_sentiment": "Предсказанная тональность",
    "confidence": "Уверенность",
    "category": "Категория",
    "rating": "Оценка",
    "topic": "Тема",
    "cluster": "Кластер",
    "aspects_text": "Аспекты",
    "aspect": "Аспект",
    "mentions": "Упоминаний",
    "negative": "Негативных",
    "neutral": "Нейтральных",
    "positive": "Позитивных",
    "negative_share": "Доля негатива",
    "term": "Термин",
    "weight": "Вес",
    "review_count": "Количество отзывов",
    "share": "Доля",
    "count": "Количество",
    "period": "Период",
    "reviews": "Отзывов",
    "positive_share": "Доля позитива",
    "avg_confidence": "Средняя уверенность",
    "metric": "Метрика",
    "value": "Значение",
}


def sentiment_palette() -> dict[str, str]:
    return {
        "negative": "#ef4444",
        "neutral": "#f59e0b",
        "positive": "#22c55e",
    }


def render_header() -> None:
    st.title("📊 Анализ отзывов")


def render_source_status(source_name: str, has_reviews: bool, can_cancel: bool = False) -> None:
    st.markdown("### Источник данных")
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            st.info(f"Текущий источник: {source_name}")
        with right:
            if can_cancel and st.button("Отменить операцию", use_container_width=True):
                request_cancel()
                st.rerun()
            if has_reviews and st.button("Сбросить текущие данные", use_container_width=True):
                st.session_state.pop("url_reviews", None)
                st.session_state.pop("url_reviews_source", None)
                st.session_state.pop("file_reviews", None)
                st.session_state.pop("file_reviews_source", None)
                st.session_state.pop("active_input_type", None)
                st.session_state.pop("sentiment_filter_mode", None)
                st.rerun()


def render_analysis_settings() -> dict:
    st.markdown("### 1. Данные и модель")
    with st.container(border=True):
        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            method = "transformer"
            st.markdown("**Модель анализа**")
            st.caption("Используется `RuBERT tiny` как основная модель тонального анализа.")
        with col_right:
            st.markdown("**Формат файла**")
            st.caption("Минимум: `text`. Дополнительно: `label`, `date`, `category`, `rating`.")

    return {
        "method": method,
    }


def render_filters(reviews: pd.DataFrame) -> dict:
    st.markdown("### 2. Фильтры")
    with st.container(border=True):
        sentiment_modes = {
            "all": ["negative", "neutral", "positive"],
            "negative": ["negative"],
            "neutral": ["neutral"],
            "positive": ["positive"],
        }
        current_mode = st.session_state.get("sentiment_filter_mode", "all")
        st.caption("Быстрый просмотр по тональности")
        buttons = st.columns(4)
        for index, (mode, title) in enumerate(
            [
                ("all", "Все"),
                ("negative", "Негативные"),
                ("neutral", "Нейтральные"),
                ("positive", "Позитивные"),
            ]
        ):
            with buttons[index]:
                if st.button(
                    title,
                    use_container_width=True,
                    type="primary" if current_mode == mode else "secondary",
                    key=f"sentiment_mode_{mode}",
                ):
                    st.session_state["sentiment_filter_mode"] = mode
                    st.rerun()

        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            categories = st.multiselect(
                "Категории",
                options=sorted(reviews["category"].dropna().unique().tolist()),
                default=sorted(reviews["category"].dropna().unique().tolist()),
            )
        with top_right:
            confidence_range = st.slider("Уверенность модели", 0.0, 1.0, (0.0, 1.0), 0.05)

        keyword = st.text_input("Поиск по тексту или аспектам").strip().lower()
        sentiments = sentiment_modes.get(current_mode, sentiment_modes["all"])
    return {
        "sentiments": sentiments,
        "categories": categories,
        "confidence_range": confidence_range,
        "keyword": keyword,
    }


def apply_filters(reviews: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = reviews[
        reviews["predicted_sentiment"].isin(filters["sentiments"])
        & reviews["category"].isin(filters["categories"])
        & reviews["confidence"].between(*filters["confidence_range"])
    ]
    keyword = filters["keyword"]
    if keyword:
        filtered = filtered[
            filtered["text"].str.lower().str.contains(keyword, regex=False)
            | filtered["aspects_text"].str.lower().str.contains(keyword, regex=False)
        ]
    return filtered


def render_summary_metrics(filtered: pd.DataFrame, total_reviews: int, model_name: str) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Отзывов", len(filtered), delta=f"из {total_reviews}")
    metric_columns[1].metric("Уверенность", f"{filtered['confidence'].mean():.1%}" if len(filtered) else "—")
    metric_columns[2].metric("Длина", f"{filtered['word_count'].mean():.1f} слов" if len(filtered) else "—")
    metric_columns[3].metric("Негативных", int((filtered["predicted_sentiment"] == "negative").sum()))
    metric_columns[4].metric("Модель", model_name)


def download_csv_button(data: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


def build_domain_frame(filtered: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    rows = []
    for term in terms:
        term_reviews = filtered[
            filtered["text"].str.lower().str.contains(term, regex=False)
            | filtered["aspects_text"].str.lower().str.contains(term, regex=False)
        ]
        rows.append(
            {
                "aspect": term,
                "mentions": len(term_reviews),
                "negative_share": term_reviews["predicted_sentiment"].eq("negative").mean() if len(term_reviews) else 0,
            }
        )
    return pd.DataFrame(rows)


def localize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns={column: DISPLAY_COLUMN_NAMES.get(column, column) for column in frame.columns})

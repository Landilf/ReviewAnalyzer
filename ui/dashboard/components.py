from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_helpers.pipeline import SENTIMENT_LABELS_RU


def sentiment_palette() -> dict[str, str]:
    return {
        "negative": "#ef4444",
        "neutral": "#f59e0b",
        "positive": "#22c55e",
    }


def render_header() -> None:
    st.title("📊 Анализ пользовательских отзывов")
    st.caption("Пошаговый дашборд для ВКР: от загрузки данных до отчёта и интерпретации результатов.")
    with st.container(border=True):
        st.markdown(
            "**Сценарий работы:** 1) загрузите отзывы → 2) посмотрите выводы → "
            "3) исследуйте аспекты/темы → 4) проверьте качество → 5) скачайте отчёт."
        )


def render_source_status(source_name: str, has_url_reviews: bool) -> None:
    with st.sidebar:
        st.info(f"Текущий источник: {source_name}")
        if has_url_reviews and st.button("Сбросить отзывы из ссылки"):
            st.session_state.pop("url_reviews", None)
            st.session_state.pop("url_reviews_source", None)
            st.rerun()


def render_analysis_settings() -> dict:
    with st.sidebar:
        st.header("1. Данные и модель")
        uploaded_file = st.file_uploader("Файл с отзывами", type=["csv", "xlsx", "xls"])
        
        # We focus only on RuBERT Transformer as requested
        method = "transformer"
        st.info("Используется модель: RuBERT Transformer (Sentiment-Balanced)")

        with st.expander("Дополнительные настройки"):
            use_spacy_aspects = st.checkbox("Извлекать аспекты через spaCy", value=False)
            show_dev_tools = st.checkbox("Показать инструменты разработчика", value=False)

        with st.expander("Формат файла"):
            st.markdown("Минимум: `text`. Дополнительно: `label`, `date`, `category`, `rating`.")

    return {
        "method": method,
        "use_spacy_aspects": use_spacy_aspects,
        "show_dev_tools": show_dev_tools,
        "uploaded_bytes": uploaded_file.getvalue() if uploaded_file else None,
        "uploaded_name": uploaded_file.name if uploaded_file else None,
    }


def render_filters(reviews: pd.DataFrame) -> dict:
    with st.sidebar:
        st.header("2. Фильтры")
        sentiments = st.multiselect(
            "Тональность",
            options=["negative", "neutral", "positive"],
            default=["negative", "neutral", "positive"],
            format_func=lambda value: SENTIMENT_LABELS_RU.get(value, value),
        )
        categories = st.multiselect(
            "Категории",
            options=sorted(reviews["category"].dropna().unique().tolist()),
            default=sorted(reviews["category"].dropna().unique().tolist()),
        )
        confidence_range = st.slider("Уверенность модели", 0.0, 1.0, (0.0, 1.0), 0.05)
        keyword = st.text_input("Поиск по тексту или аспектам").strip().lower()
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

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.dashboard.components import localize_columns


def _review_has_aspect(values, selected_aspect: str) -> bool:
    if isinstance(values, (list, tuple, set)):
        return selected_aspect in values
    if pd.isna(values):
        return False
    if isinstance(values, str):
        return selected_aspect in [part.strip() for part in values.split(",") if part.strip()]
    return False


def render_aspects(filtered: pd.DataFrame, aspect_stats: pd.DataFrame) -> None:
    problem_aspects = aspect_stats[aspect_stats["mentions"] >= 2].sort_values(
        ["negative_share", "mentions"],
        ascending=[False, False],
    )
    display_aspects = problem_aspects.head(20) if not problem_aspects.empty else aspect_stats.head(20)

    left, right = st.columns([1.2, 1])
    with left:
        fig = px.bar(
            display_aspects.sort_values("mentions"),
            x="mentions",
            y="aspect",
            orientation="h",
            color="negative_share",
            color_continuous_scale="Reds",
            labels={"mentions": "Упоминаний", "aspect": "Аспект", "negative_share": "Доля негатива"},
            title="Самые частые аспекты",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Проблемные зоны")
        st.dataframe(
            localize_columns(problem_aspects[["aspect", "mentions", "negative", "neutral", "positive", "negative_share"]].head(15)),
            use_container_width=True,
            hide_index=True,
        )

    selected_aspect = st.selectbox("Детализация аспекта", options=aspect_stats["aspect"].tolist() or [""])
    if selected_aspect:
        aspect_row = aspect_stats[aspect_stats["aspect"] == selected_aspect].iloc[0]
        aspect_reviews = filtered[filtered["aspects"].apply(lambda values: _review_has_aspect(values, selected_aspect))]
        card_left, card_right, card_third = st.columns(3)
        card_left.metric("Упоминаний", int(aspect_row["mentions"]))
        card_right.metric("Негативных", int(aspect_row["negative"]))
        card_third.metric("Уверенность", f"{aspect_reviews['confidence'].mean():.1%}" if len(aspect_reviews) else "—")
        st.dataframe(
            localize_columns(aspect_reviews[["review_id", "text", "predicted_sentiment", "confidence", "aspects_text"]].head(20)),
            use_container_width=True,
            hide_index=True,
        )


def render_topics(filtered: pd.DataFrame, result) -> None:
    filtered_topics = (
        filtered["topic"]
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="review_count")
    )
    if not filtered_topics.empty:
        filtered_topics["share"] = (filtered_topics["review_count"] / filtered_topics["review_count"].sum()).round(3)
    visible_topics = set(filtered_topics["topic"].tolist())
    visible_topic_terms = result.topic_terms[result.topic_terms["topic"].isin(visible_topics)] if not result.topic_terms.empty else result.topic_terms

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("#### Размер тем")
        st.dataframe(localize_columns(filtered_topics), use_container_width=True, hide_index=True)
        if not filtered_topics.empty:
            fig = px.pie(filtered_topics, names="topic", values="review_count", title="Распределение тем")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("После применения фильтров не осталось отзывов для тематического анализа.")

    with right:
        st.markdown("#### Ключевые слова")
        if visible_topic_terms.empty:
            st.info("После применения фильтров не осталось тем с доступными ключевыми словами.")
        else:
            fig = px.bar(
                visible_topic_terms,
                x="weight",
                y="term",
                color="topic",
                facet_col="topic",
                facet_col_wrap=2,
                orientation="h",
                title="Топ слов по темам",
            )
            fig.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig, use_container_width=True)

    selected_topic = st.selectbox("Отзывы выбранной темы", options=sorted(filtered["topic"].dropna().unique().tolist()) or [""])
    if selected_topic:
        topic_reviews = filtered[filtered["topic"] == selected_topic]
        st.dataframe(
            localize_columns(topic_reviews[["review_id", "text", "predicted_sentiment", "confidence", "topic"]].head(30)),
            use_container_width=True,
            hide_index=True,
        )


def render_clusters(filtered: pd.DataFrame) -> None:
    cluster_counts = filtered["cluster"].value_counts().rename_axis("cluster").reset_index(name="count")
    if cluster_counts.empty:
        st.info("После фильтрации не осталось отзывов для кластеризации.")
        return

    fig = px.bar(cluster_counts, x="cluster", y="count", title="Размеры кластеров")
    st.plotly_chart(fig, use_container_width=True)
    selected_cluster = st.selectbox("Отзывы выбранного кластера", options=cluster_counts["cluster"].tolist())
    cluster_reviews = filtered[filtered["cluster"] == selected_cluster]
    st.dataframe(
        localize_columns(cluster_reviews[["review_id", "text", "predicted_sentiment", "confidence", "cluster", "aspects_text"]].head(30)),
        use_container_width=True,
        hide_index=True,
    )


def render_periods(filtered: pd.DataFrame) -> None:
    if not filtered["date"].notna().any():
        st.info("В данных нет корректных дат для сравнения периодов.")
        return

    min_date = filtered["date"].min().date()
    max_date = filtered["date"].max().date()
    midpoint = min_date + (max_date - min_date) / 2
    period_left, period_right = st.columns(2)
    with period_left:
        first_range = st.date_input("Период 1", value=(min_date, midpoint), min_value=min_date, max_value=max_date)
    with period_right:
        second_range = st.date_input("Период 2", value=(midpoint, max_date), min_value=min_date, max_value=max_date)

    if len(first_range) != 2 or len(second_range) != 2:
        return

    period_frame = _compare_periods(filtered, first_range, second_range)
    st.dataframe(localize_columns(period_frame), use_container_width=True, hide_index=True)
    fig = px.bar(
        period_frame,
        x="period",
        y=["negative_share", "positive_share", "avg_confidence"],
        barmode="group",
        title="Сравнение тональности и уверенности",
    )
    st.plotly_chart(fig, use_container_width=True)


def _compare_periods(frame: pd.DataFrame, first_range, second_range) -> pd.DataFrame:
    first_start, first_end = pd.to_datetime(first_range[0]), pd.to_datetime(first_range[1])
    second_start, second_end = pd.to_datetime(second_range[0]), pd.to_datetime(second_range[1])
    periods = {
        "Период 1": frame[frame["date"].between(first_start, first_end)],
        "Период 2": frame[frame["date"].between(second_start, second_end)],
    }
    return pd.DataFrame(
        [
            {
                "period": period_name,
                "reviews": len(period_frame),
                "negative_share": period_frame["predicted_sentiment"].eq("negative").mean() if len(period_frame) else 0,
                "positive_share": period_frame["predicted_sentiment"].eq("positive").mean() if len(period_frame) else 0,
                "avg_confidence": period_frame["confidence"].mean() if len(period_frame) else 0,
            }
            for period_name, period_frame in periods.items()
        ]
    )

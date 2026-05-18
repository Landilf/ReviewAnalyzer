from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


def render_aspects(filtered: pd.DataFrame, aspect_stats: pd.DataFrame) -> None:
    left, right = st.columns([1.2, 1])
    with left:
        top_aspects = aspect_stats.head(20)
        fig = px.bar(
            top_aspects.sort_values("mentions"),
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
        problem_aspects = aspect_stats[aspect_stats["mentions"] >= 2].sort_values(
            ["negative_share", "mentions"],
            ascending=[False, False],
        )
        st.markdown("#### Проблемные зоны")
        st.dataframe(
            problem_aspects[["aspect", "mentions", "negative", "neutral", "positive", "negative_share"]].head(15),
            use_container_width=True,
            hide_index=True,
        )

    selected_aspect = st.selectbox("Детализация аспекта", options=aspect_stats["aspect"].tolist() or [""])
    if selected_aspect:
        aspect_reviews = filtered[filtered["aspects"].apply(lambda values: selected_aspect in values)]
        card_left, card_right, card_third = st.columns(3)
        card_left.metric("Упоминаний", len(aspect_reviews))
        card_right.metric("Негативных", int(aspect_reviews["predicted_sentiment"].eq("negative").sum()))
        card_third.metric("Уверенность", f"{aspect_reviews['confidence'].mean():.1%}" if len(aspect_reviews) else "—")
        st.dataframe(
            aspect_reviews[["review_id", "text", "predicted_sentiment", "confidence", "aspects_text"]].head(20),
            use_container_width=True,
            hide_index=True,
        )


def render_topics(filtered: pd.DataFrame, result) -> None:
    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("#### Размер тем")
        st.dataframe(result.topics, use_container_width=True, hide_index=True)
        if not result.topics.empty:
            fig = px.pie(result.topics, names="topic", values="review_count", title="Распределение тем")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Ключевые слова")
        if result.topic_terms.empty:
            st.info("Недостаточно данных для тематического моделирования.")
        else:
            fig = px.bar(
                result.topic_terms,
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
            topic_reviews[["review_id", "text", "predicted_sentiment", "confidence", "topic"]].head(30),
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
        cluster_reviews[["review_id", "text", "predicted_sentiment", "confidence", "cluster", "aspects_text"]].head(30),
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
    st.dataframe(period_frame, use_container_width=True, hide_index=True)
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

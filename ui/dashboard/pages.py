from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app_logger import get_logger
from analysis_helpers.insights import build_html_report
from review_parser import build_reviews_from_text, fetch_reviews_from_url
from ui.dashboard.components import (
    build_domain_frame,
    download_csv_button,
    sentiment_palette,
)
from ui.dashboard.research import render_aspects, render_clusters, render_periods, render_topics
from ui.dashboard.services import run_analysis, run_analysis_from_frame


logger = get_logger("ui.pages")

def render_brief_overview(filtered: pd.DataFrame, insights: list[str], recommendations: list[str]) -> None:
    st.subheader("Краткий обзор")
    left, right = st.columns([1.1, 0.9])

    with left:
        sentiment_share = filtered["predicted_sentiment"].value_counts(normalize=True) if len(filtered) else pd.Series(dtype=float)
        positive_share = float(sentiment_share.get("positive", 0.0))
        negative_share = float(sentiment_share.get("negative", 0.0))
        neutral_share = float(sentiment_share.get("neutral", 0.0))
        confidence = float(filtered["confidence"].mean()) if len(filtered) else 0.0

        score = max(0, min(100, int(round(50 + (positive_share - negative_share) * 50 + (confidence - 0.5) * 20))))
        st.metric("Интегральная оценка товара", f"{score}/100")
        st.caption("Оценка основана на доле позитивных/негативных отзывов и средней уверенности модели.")

        # Marketplace-like rating: weighted sentiment converted to 1..5 scale.
        raw_rating = positive_share * 5 + neutral_share * 3 + negative_share * 1
        stars_rating = round(max(1.0, min(5.0, raw_rating)), 1)
        full_stars = int(stars_rating)
        half_star = 1 if (stars_rating - full_stars) >= 0.5 else 0
        empty_stars = max(0, 5 - full_stars - half_star)
        stars_text = "★" * full_stars + ("⯨" if half_star else "") + "☆" * empty_stars
        st.metric("Оценка по 5-балльной шкале", f"{stars_rating:.1f} / 5")
        st.caption(stars_text)

        if score >= 75:
            st.success("Общий фон положительный: товар выглядит надёжным по отзывам.")
        elif score >= 50:
            st.warning("Смешанный фон: есть спорные аспекты, стоит смотреть детали.")
        else:
            st.error("Рискованный фон: в отзывах заметный перевес негативных сигналов.")

        st.markdown("#### Распределение тональности")
        sentiment_counts = filtered["predicted_sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.bar(
            sentiment_counts,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map=sentiment_palette(),
            labels={"sentiment": "Тональность", "count": "Количество"},
            title="Тональность отзывов",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Главные тезисы")
        for insight in insights[:5]:
            st.info(insight)
        st.markdown("#### Что улучшать в первую очередь")
        for recommendation in recommendations[:4]:
            st.warning(recommendation)


def render_loader() -> None:
    st.subheader("Загрузка отзывов по ссылке")
    st.markdown(
        "Вставьте ссылку на страницу товара. Приложение попробует найти отзывы на странице, "
        "сохранит их как текущий набор данных и сразу отправит в общий анализ."
    )
    with st.container(border=True):
        url = st.text_input(
            "Ссылка на товар",
            placeholder="https://otzovik.com/reviews/... или https://otzovik.com/review_123.html",
        )
        limit = st.slider("Максимум отзывов", 10, 300, 100, 10)
        use_browser = st.checkbox(
            "Использовать браузерную загрузку при блокировке",
            value=True,
            help="Playwright запускает Chromium и лучше работает с сайтами, где HTML формируется JavaScript.",
        )
        headless = st.checkbox(
            "Скрытый режим (Headless)",
            value=False,
            help="Если выключить, откроется окно браузера. Это помогает пройти капчу или увидеть причину блокировки.",
        )
        parse_clicked = st.button("Загрузить и проанализировать", type="primary")

    if parse_clicked:
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Запуск загрузки отзывов")

        def on_progress(value: float, message: str) -> None:
            percent = int(max(0, min(value, 1)) * 100)
            progress_bar.progress(percent, text=message)

        logger.info("URL import requested: url=%s limit=%s use_browser=%s headless=%s", url, limit, use_browser, headless)
        try:
            scrape_result = fetch_reviews_from_url(
                url,
                limit=limit,
                use_browser=use_browser,
                progress_callback=on_progress,
                headless=headless,
            )
        except Exception as exc:
            logger.exception("URL import failed: %s", exc)
            progress_placeholder.empty()
            st.error(f"Не удалось загрузить отзывы: {exc}")
            st.info(
                "Если сайт не отдаёт отзывы в HTML, попробуйте экспортировать отзывы в CSV "
                "или использовать другой товар. DNS/Ozon могут загружать отзывы динамически."
            )
            return

        st.session_state["url_reviews"] = scrape_result.reviews
        st.session_state["url_reviews_source"] = scrape_result.source
        logger.info("URL import succeeded: rows=%s source=%s", len(scrape_result.reviews), scrape_result.source)
        progress_bar.progress(100, text=f"Готово: загружено {len(scrape_result.reviews)} отзывов")
        
        if scrape_result.warning:
            st.warning(scrape_result.warning)
            
        st.success(scrape_result.message)
        st.rerun()

    st.divider()
    st.subheader("Запасной вариант: вставка отзывов текстом")
    st.markdown("Если сайт блокирует автоматическую загрузку, скопируйте отзывы вручную и вставьте их ниже.")
    manual_text = st.text_area(
        "Отзывы",
        placeholder="Один отзыв — один абзац. Между отзывами оставляйте пустую строку.",
        height=180,
    )
    if st.button("Использовать вставленные отзывы"):
        logger.info("Manual import requested: text_length=%s", len(manual_text))
        manual_reviews = build_reviews_from_text(manual_text)
        if manual_reviews.empty:
            logger.warning("Manual import produced empty dataset")
            st.error("Не удалось выделить отзывы из текста. Разделяйте отзывы пустыми строками.")
        else:
            st.session_state["url_reviews"] = manual_reviews
            st.session_state["url_reviews_source"] = "ручная вставка"
            logger.info("Manual import succeeded: rows=%s", len(manual_reviews))
            st.success(f"Загружено {len(manual_reviews)} отзывов из вставленного текста.")
            st.rerun()

    if "url_reviews" in st.session_state:
        st.success(f"Сейчас используются отзывы из ссылки: {st.session_state.get('url_reviews_source', 'неизвестный источник')}")
        st.dataframe(st.session_state["url_reviews"].head(20), use_container_width=True, hide_index=True)


def render_home(filtered: pd.DataFrame, insights: list[str], recommendations: list[str]) -> None:
    st.subheader("Краткая картина")
    left, right = st.columns([1, 1])

    with left:
        sentiment_counts = filtered["predicted_sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
        fig = px.bar(
            sentiment_counts,
            x="sentiment",
            y="count",
            color="sentiment",
            color_discrete_map=sentiment_palette(),
            labels={"sentiment": "Тональность", "count": "Количество"},
            title="Распределение тональности",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Автоматические выводы")
        for insight in insights:
            st.success(insight)
        st.markdown("#### Рекомендации")
        for recommendation in recommendations:
            st.warning(recommendation)

    with st.expander("Динамика и оценки", expanded=False):
        _render_timeline(filtered)
        if "rating" in filtered.columns:
            fig = px.box(
                filtered,
                x="predicted_sentiment",
                y="rating",
                color="predicted_sentiment",
                color_discrete_map=sentiment_palette(),
                title="Оценки по тональности",
                labels={"rating": "Оценка", "predicted_sentiment": "Тональность"},
            )
            st.plotly_chart(fig, use_container_width=True)


def render_research(filtered: pd.DataFrame, result, aspect_stats: pd.DataFrame) -> None:
    st.subheader("Исследование данных")
    mode = st.radio(
        "Что изучаем?",
        ["Аспекты", "Темы", "Кластеры", "Периоды"],
        index=0,
        horizontal=True,
    )

    if mode == "Аспекты":
        render_aspects(filtered, aspect_stats)
    elif mode == "Темы":
        render_topics(filtered, result)
    elif mode == "Кластеры":
        render_clusters(filtered)
    elif mode == "Периоды":
        render_periods(filtered)


def render_reviews(filtered: pd.DataFrame, all_reviews: pd.DataFrame) -> None:
    st.subheader("Отзывы и экспорт таблиц")
    visible_columns = [
        "review_id",
        "text",
        "label",
        "predicted_sentiment",
        "confidence",
        "category",
        "rating",
        "topic",
        "cluster",
        "aspects_text",
    ]
    existing_columns = [column for column in visible_columns if column in filtered.columns]
    st.dataframe(filtered[existing_columns], use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        download_csv_button(filtered, "Скачать отфильтрованные отзывы", "review_analysis_filtered.csv")
    with right:
        download_csv_button(all_reviews, "Скачать полный результат", "review_analysis_full.csv")


def render_quality(result, options: dict) -> None:
    st.subheader("Качество модели")
    if result.metrics:
        metrics_frame = pd.DataFrame(
            [{"metric": metric, "value": value} for metric, value in result.metrics.items()]
        )
        fig = px.bar(
            metrics_frame,
            x="metric",
            y="value",
            text="value",
            title="Метрики классификации",
            labels={"metric": "Метрика", "value": "Значение"},
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
    else:
        st.info("Метрики качества доступны, если в данных есть столбец `label`.")

    if not result.confusion.empty:
        st.markdown("#### Матрица ошибок")
        fig = px.imshow(
            result.confusion,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Предсказание", "y": "Истинная метка", "color": "Количество"},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_report(filtered: pd.DataFrame, aspect_stats: pd.DataFrame, result, insights: list[str], recommendations: list[str]) -> None:
    st.subheader("Отчёт и предметный словарь")
    domain_terms = st.text_area(
        "Важные аспекты предметной области",
        value="доставка\nкачество\nцена\nсервис\nупаковка\nбатарея\nэкран",
        help="Один аспект на строку. Таблица покажет, насколько часто они встречаются и насколько проблемны.",
    )
    selected_terms = [term.strip().lower() for term in domain_terms.splitlines() if term.strip()]
    st.dataframe(build_domain_frame(filtered, selected_terms), use_container_width=True, hide_index=True)

    html_report = build_html_report(filtered, aspect_stats, result.topics, insights, recommendations)
    st.download_button(
        "Скачать HTML-отчёт",
        data=html_report.encode("utf-8"),
        file_name="review_analysis_report.html",
        mime="text/html",
    )


def _render_timeline(filtered: pd.DataFrame) -> None:
    if not filtered["date"].notna().any():
        st.info("В данных нет корректных дат для динамики.")
        return

    timeline = (
        filtered.dropna(subset=["date"])
        .assign(date=lambda frame: frame["date"].dt.date)
        .groupby(["date", "predicted_sentiment"], as_index=False)
        .size()
    )
    fig = px.area(
        timeline,
        x="date",
        y="size",
        color="predicted_sentiment",
        color_discrete_map=sentiment_palette(),
        labels={"date": "Дата", "size": "Количество", "predicted_sentiment": "Тональность"},
        title="Динамика отзывов во времени",
    )
    st.plotly_chart(fig, use_container_width=True)

from __future__ import annotations

import io
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app_control import is_cancel_requested, set_operation_running
from app_logger import get_logger
from analysis_helpers.insights import build_html_report
from review_parser import build_reviews_from_text, fetch_reviews_from_url
from ui.dashboard.components import (
    build_domain_frame,
    download_csv_button,
    localize_columns,
    sentiment_palette,
)
from ui.dashboard.research import render_aspects, render_clusters, render_periods, render_topics


logger = get_logger("ui.pages")


def process_pending_url_import() -> None:
    pending = st.session_state.get("pending_url_import")
    if not pending:
        return

    url = pending["url"]
    use_browser = pending["use_browser"]
    headless = pending["headless"]
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Запуск загрузки отзывов")

    def on_progress(value: float, message: str) -> None:
        percent = int(max(0, min(value, 1)) * 100)
        progress_bar.progress(percent, text=message)

    logger.info("URL import requested: url=%s use_browser=%s headless=%s", url, use_browser, headless)
    try:
        scrape_result = fetch_reviews_from_url(
            url,
            use_browser=use_browser,
            progress_callback=on_progress,
            headless=headless,
            cancel_check=is_cancel_requested,
        )
    except RuntimeError as exc:
        set_operation_running(False)
        progress_placeholder.empty()
        st.session_state.pop("pending_url_import", None)
        st.session_state.pop("cancel_requested", None)
        st.warning(str(exc))
        st.stop()
    except Exception as exc:
        set_operation_running(False)
        logger.exception("URL import failed: %s", exc)
        progress_placeholder.empty()
        st.session_state.pop("pending_url_import", None)
        st.error(f"Не удалось загрузить отзывы: {exc}")
        st.info(
            "Если сайт не отдаёт отзывы в HTML, попробуйте экспортировать отзывы в CSV "
            "или использовать другой товар. Некоторые сайты могут загружать отзывы динамически."
        )
        st.stop()

    st.session_state["url_reviews"] = scrape_result.reviews
    st.session_state["url_reviews_source"] = scrape_result.source
    st.session_state["active_input_type"] = "url"
    st.session_state["url_reviews_meta"] = {
        "source": scrape_result.source,
        "label": "url-import",
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "rows": int(len(scrape_result.reviews)),
        "columns": list(scrape_result.reviews.columns),
    }
    logger.info("URL import succeeded: rows=%s source=%s", len(scrape_result.reviews), scrape_result.source)
    progress_bar.progress(100, text=f"Готово: загружено {len(scrape_result.reviews)} отзывов")
    set_operation_running(False)
    st.session_state.pop("pending_url_import", None)

    if scrape_result.warning:
        st.warning(scrape_result.warning)

    st.success(scrape_result.message)
    st.session_state.pop("cancel_requested", None)
    st.rerun()

def render_brief_overview(filtered: pd.DataFrame, insights: list[str], recommendations: list[str]) -> None:
    st.subheader("Краткий обзор")
    if filtered.empty:
        st.info("После применения текущих фильтров не осталось отзывов для построения краткого обзора.")
        return

    left, right = st.columns([1.1, 0.9])

    with left:
        sentiment_share = filtered["predicted_sentiment"].value_counts(normalize=True)
        positive_share = float(sentiment_share.get("positive", 0.0))
        negative_share = float(sentiment_share.get("negative", 0.0))
        neutral_share = float(sentiment_share.get("neutral", 0.0))
        confidence = float(filtered["confidence"].mean())

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

        st.markdown("#### Динамика отзывов")
        _render_timeline(filtered)

    with right:
        st.markdown("#### Главные тезисы")
        for insight in insights[:5]:
            st.info(insight)
        st.markdown("#### Что улучшать в первую очередь")
        for recommendation in recommendations[:4]:
            st.warning(recommendation)


def render_loader() -> None:
    st.subheader("Входные данные")
    tabs = st.tabs(["Ссылка", "Файл", "Текст"])

    with tabs[0]:
        st.markdown(
            "Вставьте ссылку на страницу товара. Приложение попробует найти отзывы на странице, "
            "сохранит их как текущий набор данных и сразу отправит в общий анализ."
        )
        with st.container(border=True):
            url = st.text_input(
                "Ссылка на товар",
                placeholder="https://otzovik.com/reviews/... или https://otzovik.com/review_123.html",
                key="review_url_input",
            )
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
            parse_clicked = st.button("Загрузить и проанализировать", type="primary", key="url_import_button")

        if parse_clicked:
            st.session_state["pending_url_import"] = {
                "url": url,
                "use_browser": use_browser,
                "headless": headless,
            }
            st.session_state.pop("cancel_requested", None)
            set_operation_running(True)
            st.rerun()

        if "url_reviews" in st.session_state:
            st.success(f"Сейчас используются отзывы из ссылки: {st.session_state.get('url_reviews_source', 'неизвестный источник')}")
            st.dataframe(localize_columns(st.session_state["url_reviews"].head(20)), use_container_width=True, hide_index=True)
            _render_review_downloads(
                st.session_state["url_reviews"],
                st.session_state.get("url_reviews_source", "unknown"),
                st.session_state.get("url_reviews_meta", {}),
            )

    with tabs[1]:
        st.markdown("Загрузите CSV или Excel-файл с отзывами — анализ начнётся сразу после выбора файла.")
        uploaded_file = st.file_uploader("Файл с отзывами", type=["csv", "xlsx", "xls"], key="input_file_uploader")
        if uploaded_file is not None:
            st.caption(f"Выбран файл: `{uploaded_file.name}`")
            try:
                preview_df = st.session_state.get("file_reviews")
                current_source = st.session_state.get("file_reviews_source")
                if preview_df is None or current_source != uploaded_file.name:
                    preview_df = _read_uploaded_reviews(uploaded_file)
                    st.session_state["file_reviews"] = preview_df
                    st.session_state["file_reviews_source"] = uploaded_file.name
                    st.session_state["file_reviews_meta"] = {
                        "source": "file-upload",
                        "label": "file-import",
                        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        "rows": int(len(preview_df)),
                        "columns": list(preview_df.columns),
                    }
                    st.session_state["active_input_type"] = "file"
                    st.session_state.pop("cancel_requested", None)
                    st.success(f"Загружено {len(preview_df)} отзывов из файла.")
            except Exception as exc:
                st.error(f"Не удалось прочитать файл: {exc}")
                preview_df = None

            if preview_df is not None:
                st.dataframe(localize_columns(preview_df.head(20)), use_container_width=True, hide_index=True)
                _render_review_downloads(
                    preview_df,
                    uploaded_file.name,
                    st.session_state.get("file_reviews_meta", {}),
                )
        else:
            st.info("Сначала выберите файл — он будет использован автоматически.")

        if "file_reviews" in st.session_state and st.session_state.get("file_reviews_source"):
            st.success(f"Сейчас используются отзывы из файла: {st.session_state.get('file_reviews_source', 'неизвестный файл')}")

    with tabs[2]:
        st.markdown("Если сайт блокирует автоматическую загрузку, скопируйте отзывы вручную и вставьте их ниже.")
        manual_text = st.text_area(
            "Отзывы",
            placeholder="Один отзыв — один абзац. Между отзывами оставляйте пустую строку.",
            height=180,
            key="manual_reviews_input",
        )
        if st.button("Использовать вставленные отзывы", key="manual_import_button"):
            logger.info("Manual import requested: text_length=%s", len(manual_text))
            manual_reviews = build_reviews_from_text(manual_text)
            if manual_reviews.empty:
                logger.warning("Manual import produced empty dataset")
                st.error("Не удалось выделить отзывы из текста. Разделяйте отзывы пустыми строками.")
            else:
                st.session_state["file_reviews"] = manual_reviews
                st.session_state["file_reviews_source"] = "ручная вставка"
                st.session_state["active_input_type"] = "manual"
                st.session_state["file_reviews_meta"] = {
                    "source": "manual-input",
                    "label": "manual-import",
                    "created_at": pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S"),
                    "rows": int(len(manual_reviews)),
                    "columns": list(manual_reviews.columns),
                }
                st.session_state.pop("cancel_requested", None)
                logger.info("Manual import succeeded: rows=%s", len(manual_reviews))
                st.success(f"Загружено {len(manual_reviews)} отзывов из вставленного текста.")
                st.rerun()

        if "file_reviews" in st.session_state and st.session_state.get("file_reviews_source") == "ручная вставка":
            st.success("Сейчас используются отзывы из ручной вставки.")
            st.dataframe(localize_columns(st.session_state["file_reviews"].head(20)), use_container_width=True, hide_index=True)
            _render_review_downloads(
                st.session_state["file_reviews"],
                st.session_state.get("file_reviews_source", "unknown"),
                st.session_state.get("file_reviews_meta", {}),
            )


def _render_review_downloads(reviews: pd.DataFrame, source_name: str, metadata: dict) -> None:
    st.markdown("#### Скачать распарсенные отзывы")
    left, right = st.columns(2)
    csv_file_name = f"parsed_reviews_{source_name.replace(' ', '_').replace('/', '_')}.csv"
    json_file_name = f"parsed_reviews_{source_name.replace(' ', '_').replace('/', '_')}.json"

    csv_bytes = reviews.to_csv(index=False).encode("utf-8-sig")
    json_payload = json.dumps(
        {
            **metadata,
            "source": source_name,
            "reviews": reviews.to_dict(orient="records"),
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")

    with left:
        st.download_button(
            "Скачать CSV",
            data=csv_bytes,
            file_name=csv_file_name,
            mime="text/csv",
        )
    with right:
        st.download_button(
            "Скачать JSON",
            data=json_payload,
            file_name=json_file_name,
            mime="application/json",
        )


def _read_uploaded_reviews(uploaded_file) -> pd.DataFrame:
    buffer = io.BytesIO(uploaded_file.getvalue())
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


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
    st.dataframe(localize_columns(filtered[existing_columns]), use_container_width=True, hide_index=True)

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
        st.dataframe(localize_columns(metrics_frame), use_container_width=True, hide_index=True)
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


def render_report(
    filtered: pd.DataFrame,
    aspect_stats: pd.DataFrame,
    result,
    insights: list[str],
    recommendations: list[str],
    *,
    source_name: str,
    filters: dict,
    options: dict,
) -> None:
    st.subheader("Отчёт и предметный словарь")
    filtered_topics = (
        filtered["topic"]
        .value_counts()
        .rename_axis("topic")
        .reset_index(name="review_count")
    )
    if not filtered_topics.empty:
        filtered_topics["share"] = (filtered_topics["review_count"] / filtered_topics["review_count"].sum()).round(3)

    domain_terms = st.text_area(
        "Важные аспекты предметной области",
        value="доставка\nкачество\nцена\nсервис\nупаковка\nбатарея\nэкран",
        help="Один аспект на строку. Таблица покажет, насколько часто они встречаются и насколько проблемны.",
    )
    selected_terms = [term.strip().lower() for term in domain_terms.splitlines() if term.strip()]
    st.dataframe(localize_columns(build_domain_frame(filtered, selected_terms)), use_container_width=True, hide_index=True)

    html_report = build_html_report(
        filtered,
        aspect_stats,
        filtered_topics,
        insights,
        recommendations,
        model_name=result.model_name,
        model_metrics=result.metrics,
        confusion=result.confusion,
        source_name=source_name,
        filters=filters,
        options=options,
    )
    model_slug = result.model_name.lower().replace(" ", "_").replace("/", "_")
    st.download_button(
        "Скачать HTML-отчёт",
        data=html_report.encode("utf-8"),
        file_name=f"review_analysis_report_{model_slug}.html",
        mime="text/html",
    )


def _render_timeline(filtered: pd.DataFrame) -> None:
    if not filtered["date"].notna().any():
        st.info("В данных нет корректных дат для динамики.")
        return

    timeline = (
        filtered.dropna(subset=["date"])
        .assign(date=lambda frame: frame["date"].dt.date)
        .groupby("date", as_index=False)
        .size()
    )
    fig = px.bar(
        timeline,
        x="date",
        y="size",
        labels={"date": "Дата", "size": "Количество"},
        title="Динамика количества отзывов во времени",
    )
    st.plotly_chart(fig, use_container_width=True)

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    print("Этот файл нужно запускать через Streamlit:")
    print("  streamlit run streamlit_app.py")
    raise SystemExit(1)

from app_logger import get_logger, setup_logging
from analysis_helpers.insights import build_insights, build_recommendations
from analysis_helpers.pipeline import build_aspect_stats
from ui.dashboard.components import (
    apply_filters,
    render_analysis_settings,
    render_filters,
    render_header,
    render_source_status,
    render_summary_metrics,
)
from ui.dashboard.developer import render_developer_panel
from ui.dashboard.pages import (
    render_brief_overview,
    render_home,
    render_loader,
    render_quality,
    render_report,
    render_research,
    render_reviews,
)
from ui.dashboard.services import run_analysis, run_analysis_from_frame


st.set_page_config(
    page_title="Анализ и визуализация отзывов",
    page_icon="📊",
    layout="wide",
)

logger = setup_logging()


def main() -> None:
    logger.info("Streamlit app started")
    render_header()
    render_loader()
    options = render_analysis_settings()
    logger.info(
        "Analysis settings: method=%s use_spacy_aspects=%s has_uploaded_file=%s",
        options["method"],
        options["use_spacy_aspects"],
        bool(options["uploaded_bytes"]),
    )
    has_url_reviews = "url_reviews" in st.session_state
    source_name = st.session_state.get("url_reviews_source") if has_url_reviews else "файл или пример reviews.csv"
    render_source_status(source_name, has_url_reviews)
    active_frame = st.session_state.get("url_reviews") if has_url_reviews else None
    options["active_frame"] = active_frame

    # Always use RuBERT transformer
    active_method = "transformer"
    options["method"] = active_method

    try:
        if has_url_reviews:
            logger.info("Running analysis from URL/manual reviews: rows=%s", len(active_frame))
            result = run_analysis_from_frame(
                active_frame,
                active_method,
                options["use_spacy_aspects"],
            )
        else:
            logger.info("Running analysis from file/default dataset")
            result = run_analysis(
                options["uploaded_bytes"],
                options["uploaded_name"],
                active_method,
                options["use_spacy_aspects"],
            )
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        st.error(f"Не удалось выполнить анализ: {exc}")
        st.stop()

    filters = render_filters(result.reviews)
    logger.info("Filters: %s", filters)
    filtered = apply_filters(result.reviews, filters)
    logger.info("Filtered reviews: %s of %s", len(filtered), len(result.reviews))
    aspect_stats = build_aspect_stats(filtered) if not filtered.empty else result.aspect_stats.iloc[0:0]
    insights = build_insights(filtered, aspect_stats)
    recommendations = build_recommendations(aspect_stats)

    render_summary_metrics(filtered, len(result.reviews), result.model_name)
    tabs = st.tabs(["Краткий обзор", "Расширенный анализ"])

    with tabs[0]:
        render_brief_overview(filtered, insights, recommendations)
        with st.expander("Показать дополнительный обзор"):
            render_home(filtered, insights, recommendations)

    with tabs[1]:
        advanced_tabs = st.tabs(["Исследование", "Отзывы", "Качество", "Отчёт"])
        with advanced_tabs[0]:
            render_research(filtered, result, aspect_stats)
        with advanced_tabs[1]:
            render_reviews(filtered, result.reviews)
        with advanced_tabs[2]:
            render_quality(result, options)
        with advanced_tabs[3]:
            render_report(filtered, aspect_stats, result, insights, recommendations)
        if options.get("show_dev_tools", False):
            with st.expander("Инструменты разработчика", expanded=False):
                render_developer_panel(result.reviews)

    st.caption("Подсказка для защиты: покажи фильтрацию негативных отзывов по аспекту и экспорт результата.")


main()

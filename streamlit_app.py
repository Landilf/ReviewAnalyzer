import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    print("Этот файл нужно запускать через Streamlit:")
    print("  streamlit run streamlit_app.py")
    raise SystemExit(1)

from app_logger import get_logger, setup_logging
from app_control import clear_cancel, is_cancel_requested, is_operation_running, set_operation_running
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
from ui.dashboard.pages import (
    process_pending_url_import,
    render_brief_overview,
    render_loader,
    render_quality,
    render_report,
    render_research,
    render_reviews,
)
from ui.dashboard.services import run_analysis_from_frame


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
        "Analysis settings: method=%s model=RuBERT tiny",
        options["method"],
    )
    active_input_type = st.session_state.get("active_input_type")
    has_url_reviews = active_input_type == "url" and "url_reviews" in st.session_state
    has_file_reviews = active_input_type in {"file", "manual"} and "file_reviews" in st.session_state
    has_reviews = has_url_reviews or has_file_reviews
    source_name = (
        st.session_state.get("url_reviews_source")
        if has_url_reviews
        else (st.session_state.get("file_reviews_source") if has_file_reviews else "источник не выбран")
    )
    render_source_status(source_name, has_reviews, can_cancel=is_operation_running() and not is_cancel_requested())
    process_pending_url_import()

    active_input_type = st.session_state.get("active_input_type")
    has_url_reviews = active_input_type == "url" and "url_reviews" in st.session_state
    has_file_reviews = active_input_type in {"file", "manual"} and "file_reviews" in st.session_state
    has_reviews = has_url_reviews or has_file_reviews
    active_frame = st.session_state.get("url_reviews") if has_url_reviews else st.session_state.get("file_reviews")
    options["active_frame"] = active_frame

    # Always use RuBERT transformer
    active_method = "transformer"
    options["method"] = active_method

    if not has_reviews:
        st.info("Чтобы запустить анализ, выберите файл, импортируйте отзывы по ссылке или вставьте текст вручную.")
        st.stop()

    try:
        set_operation_running(True)
        if has_url_reviews:
            logger.info("Running analysis from URL/manual reviews: rows=%s", len(active_frame))
            result = run_analysis_from_frame(
                active_frame,
                active_method,
                cancel_check=is_cancel_requested,
            )
        else:
            logger.info("Running analysis from file/manual reviews: rows=%s", len(active_frame))
            result = run_analysis_from_frame(
                active_frame,
                active_method,
                cancel_check=is_cancel_requested,
            )
    except RuntimeError as exc:
        logger.warning("Operation cancelled: %s", exc)
        clear_cancel()
        st.warning(str(exc))
        st.stop()
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        st.error(f"Не удалось выполнить анализ: {exc}")
        st.stop()
    finally:
        set_operation_running(False)
        if not is_cancel_requested():
            clear_cancel()

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

    with tabs[1]:
        advanced_tabs = st.tabs(["Исследование", "Отзывы", "Качество", "Отчёт"])
        with advanced_tabs[0]:
            render_research(filtered, result, aspect_stats)
        with advanced_tabs[1]:
            render_reviews(filtered, result.reviews)
        with advanced_tabs[2]:
            render_quality(result, options)
        with advanced_tabs[3]:
            render_report(
                filtered,
                aspect_stats,
                result,
                insights,
                recommendations,
                source_name=source_name,
                filters=filters,
                options=options,
            )
main()

import streamlit as st
import time
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
from ui.dashboard.services import run_advanced_analysis_for_result, run_analysis_from_frame
from ui.dashboard.training import render_training_sidebar


st.set_page_config(
    page_title="Анализ и визуализация отзывов",
    page_icon="📊",
    layout="wide",
)

logger = setup_logging()


def main() -> None:
    logger.info("Streamlit app started")
    render_training_sidebar()
    render_header()
    render_loader()
    options = render_analysis_settings()
    logger.info(
        "Analysis settings: method=%s model=RuBERT tiny",
        options["method"],
    )
    active_input_type, active_frame, source_name = _get_active_source()
    has_reviews = active_frame is not None
    render_source_status(source_name, has_reviews, can_cancel=is_operation_running() and not is_cancel_requested())
    process_pending_url_import()

    active_input_type, active_frame, source_name = _get_active_source()
    has_reviews = active_frame is not None
    options["active_frame"] = active_frame

    # Always use RuBERT transformer
    active_method = "transformer"
    options["method"] = active_method

    if not has_reviews:
        st.info("Чтобы запустить анализ, выберите файл, импортируйте отзывы по ссылке или вставьте текст вручную.")
        st.stop()

    result_key = f"analysis_result_{active_input_type}"
    result = st.session_state.get(result_key)
    if result is None:
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Подготовка анализа")
        started_at = time.monotonic()

        def on_analysis_progress(value: float, message: str) -> None:
            elapsed = time.monotonic() - started_at
            eta = elapsed * (1 - value) / value if value > 0.01 else 0
            progress_bar.progress(
                int(value * 100),
                text=f"{message} · {int(value * 100)}% · прошло {_format_duration(elapsed)} · осталось {_format_duration(eta)}",
            )

        try:
            set_operation_running(True)
            logger.info("Running base analysis: source=%s rows=%s", active_input_type, len(active_frame))
            result = run_analysis_from_frame(
                active_frame,
                active_method,
                cancel_check=is_cancel_requested,
                progress_callback=on_analysis_progress,
                include_advanced=False,
            )
            st.session_state[result_key] = result
            progress_bar.progress(100, text="Основной анализ завершён")
            progress_placeholder.empty()
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
            if not result.advanced_ready:
                st.info("Темы и кластеры считаются отдельно, чтобы не задерживать основной результат.")
                if st.button("Рассчитать расширенную аналитику", type="primary"):
                    _run_advanced_analysis(result, result_key)
                    st.rerun()
            else:
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
def _get_active_source():
    source_type = st.session_state.get("active_input_type")
    if source_type not in {"url", "file", "manual"}:
        return None, None, "источник не выбран"
    reviews = st.session_state.get(f"{source_type}_reviews")
    if reviews is None:
        return source_type, None, "источник не выбран"
    return source_type, reviews, st.session_state.get(f"{source_type}_reviews_source", "неизвестный источник")


def _run_advanced_analysis(result, result_key: str) -> None:
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0, text="Подготовка расширенного анализа")
    started_at = time.monotonic()

    def on_progress(value: float, message: str) -> None:
        elapsed = time.monotonic() - started_at
        eta = elapsed * (1 - value) / value if value > 0.01 else 0
        progress_bar.progress(
            int(value * 100),
            text=f"{message} · {int(value * 100)}% · прошло {_format_duration(elapsed)} · осталось {_format_duration(eta)}",
        )

    try:
        set_operation_running(True)
        st.session_state[result_key] = run_advanced_analysis_for_result(
            result,
            cancel_check=is_cancel_requested,
            progress_callback=on_progress,
        )
    finally:
        set_operation_running(False)
        progress_placeholder.empty()


def _format_duration(seconds: float) -> str:
    total = max(round(seconds), 0)
    minutes, secs = divmod(total, 60)
    if minutes:
        return f"{minutes} мин {secs} сек"
    return f"{secs} сек"


main()

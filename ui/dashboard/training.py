"""Non-blocking page controls for local aspect-classifier fine-tuning."""

from __future__ import annotations

import csv
import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from absa_training.data import load_examples
from absa_training.registry import is_published, publish_epoch, unpublish_epoch, unpublish_run
from absa_training.trainer import TrainingCancelled, TrainingConfig, activate_epoch, fine_tune, mark_run_status, resume_fine_tune
from analysis_helpers.config import MODEL_MAX_TOKENS


DATASETS_DIR = Path("data/absa/train")
VALIDATION_DIR = Path("data/absa/validation")
TEST_DIR = Path("data/absa/test")
ARTIFACTS_DIR = Path("artifacts/absa")
JOB_STATE_KEY = "absa_training_job"


@dataclass
class TrainingJob:
    """Thread-safe state shared by the Streamlit UI and a training worker."""

    run_dir: Path
    started_at: datetime = field(default_factory=datetime.now)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    progress: float = 0.0
    message: str = "Подготовка обучения"
    status: str = "running"
    result: dict | None = None
    error: str | None = None
    thread: threading.Thread | None = None
    refreshed_after_completion: bool = False

    def update_progress(self, value: float, message: str) -> None:
        with self.lock:
            self.progress = min(max(value, 0.0), 1.0)
            self.message = message

    def finish(self, result: dict) -> None:
        with self.lock:
            self.progress = 1.0
            self.message = "Дообучение завершено"
            self.status = "completed"
            self.result = result

    def fail(self, error: Exception) -> None:
        with self.lock:
            self.message = "Обучение не завершено"
            self.status = "cancelled" if isinstance(error, TrainingCancelled) else "failed"
            self.error = str(error)

    def snapshot(self) -> tuple[float, str, str, dict | None, str | None]:
        with self.lock:
            return self.progress, self.message, self.status, self.result, self.error

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def consume_completion_refresh(self) -> bool:
        """Return True once after a terminal state so the outer sidebar can refresh."""
        with self.lock:
            if self.status == "running" or self.refreshed_after_completion:
                return False
            self.refreshed_after_completion = True
            return True


def available_training_datasets() -> list[Path]:
    return sorted(DATASETS_DIR.glob("*.csv"))


def available_evaluation_datasets(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.csv"))


@st.cache_data(show_spinner=False)
def _describe_training_dataset(path_text: str, modified_at_ns: int) -> dict[str, int]:
    """Read only lightweight selection metadata; cache is invalidated on file change."""
    del modified_at_ns
    with Path(path_text).open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "examples": len(rows),
        "reviews": len({row.get("review_id", "") for row in rows}),
        "taxonomies": len({row.get("aspect_normalized", "") for row in rows if row.get("aspect_normalized", "")}),
    }


def available_runs() -> list[Path]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    return sorted(
        (path for path in ARTIFACTS_DIR.iterdir() if path.is_dir() and (path / "run_manifest.json").is_file()),
        reverse=True,
    )


def _run_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _run_label(run_dir: Path) -> str:
    """A searchable, human-readable label instead of an opaque directory name."""
    manifest = _run_manifest(run_dir)
    sources = [Path(path).stem.removesuffix("_absa_full").removesuffix("_absa_sentiment_refined") for path in manifest.get("source_files", [])]
    source_label = ", ".join(sources[:2]) or "без источника"
    if len(sources) > 2:
        source_label += f" +{len(sources) - 2}"
    created_at = manifest.get("created_at", "")
    try:
        created_label = datetime.fromisoformat(created_at).astimezone().strftime("%d.%m.%Y %H:%M")
    except ValueError:
        created_label = run_dir.name.removeprefix("run_").replace("_", " ")
    completed = int(manifest.get("last_completed_epoch", 0))
    planned = int(manifest.get("config", {}).get("epochs", 0))
    history_path = run_dir / "history.json"
    best_score = None
    if history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
        if history:
            best_score = max(item.get("joint_macro_f1", 0.0) for item in history)
    score_label = f" · лучший F1 {best_score:.3f}" if best_score is not None else ""
    return f"{created_label} · {source_label} · {completed}/{planned} эпох{score_label}"


def _select_run(
    label: str, *, key: str, disabled: bool = False, runs: list[Path] | None = None
) -> Path | None:
    """Select one run; Streamlit selectboxes can be searched by their descriptive label."""
    runs = available_runs() if runs is None else runs
    if not runs:
        return None
    return st.selectbox(
        label,
        options=runs,
        format_func=_run_label,
        key=key,
        disabled=disabled,
        help="Начните печатать дату или имя CSV, чтобы быстро отфильтровать список.",
    )


def render_training_page() -> None:
    """Render the dedicated ABSA fine-tuning workspace."""
    job = st.session_state.get(JOB_STATE_KEY)
    running = isinstance(job, TrainingJob) and job.is_running()
    st.title("Дообучение ABSA")
    st.caption("Локальное обучение классификатора таксономии аспекта и его тональности.")
    st.caption(f"Контекст модели: {MODEL_MAX_TOKENS} токенов (общая настройка проекта).")
    if isinstance(job, TrainingJob):
        _render_job_status(job)

    training_tab, models_tab, metrics_tab, comparison_tab, epochs_tab = st.tabs(
        ["Новое обучение", "Сохранённые модели", "Метрики", "Сравнение моделей", "Управление эпохами"]
    )
    with training_tab:
        datasets = available_training_datasets()
        validation_paths = available_evaluation_datasets(VALIDATION_DIR)
        test_paths = available_evaluation_datasets(TEST_DIR)
        dataset_info = {
            path.name: _describe_training_dataset(str(path), path.stat().st_mtime_ns)
            for path in datasets
        }
        option_names = [path.name for path in datasets]
        selection_key = "absa_training_dataset_selection"
        select_all, clear_selection = st.columns(2)
        if select_all.button("Выбрать все", disabled=running, key="absa_select_all"):
            st.session_state[selection_key] = option_names
        if clear_selection.button("Очистить", disabled=running, key="absa_clear_selection"):
            st.session_state[selection_key] = []
        selected_names = st.multiselect(
            "Файлы разметки", options=option_names,
            format_func=lambda name: (
                f"{name} · {dataset_info[name]['examples']:,} аспектов · "
                f"{dataset_info[name]['reviews']:,} отзывов · {dataset_info[name]['taxonomies']} категорий"
            ),
            placeholder="Выберите один или несколько наборов", key=selection_key, disabled=running,
        )
        epochs = st.number_input("Эпохи", min_value=1, max_value=50, value=4, step=1, key="absa_training_epochs", disabled=running)
        with st.expander("Параметры обучения"):
            batch_size = st.number_input("Размер batch", min_value=1, max_value=128, value=16, step=1, key="absa_training_batch", disabled=running)
            learning_rate = st.number_input(
                "Learning rate", min_value=0.000001, max_value=0.01, value=0.00002,
                step=0.000001, format="%.6f", key="absa_training_lr", disabled=running,
            )
            if validation_paths:
                validation_fraction = 0.2
                st.caption("Используется внешний validation-набор; внутренняя валидация отключена.")
            else:
                validation_fraction = st.slider(
                    "Доля внутренней валидации", min_value=0.1, max_value=0.4, value=0.2,
                    key="absa_training_validation", disabled=running,
                )
        selected_paths = [path for path in datasets if path.name in selected_names]
        if selected_paths:
            examples = sum(dataset_info[path.name]["examples"] for path in selected_paths)
            reviews = sum(dataset_info[path.name]["reviews"] for path in selected_paths)
            st.caption(f"Выбрано: {len(selected_paths)} файл(а), {examples:,} аспектных строк, до {reviews:,} отзывов.")
        else:
            st.info("Выберите хотя бы один размеченный CSV для обучения.")
        if validation_paths:
            st.success(f"Validation: {', '.join(path.name for path in validation_paths)}. Лучшая эпоха выбирается по нему.")
        else:
            st.warning("Внешний validation-набор не найден: будет создана внутренняя валидация из train.")
        if test_paths:
            st.info(f"Test: {', '.join(path.name for path in test_paths)}. Проверка выполнится один раз на лучшей эпохе.")
        else:
            st.warning("Test-набор не найден: итоговая независимая оценка не будет рассчитана.")
        initial_checkpoint = _render_initial_model_picker(disabled=running)
        if st.button("Запустить дообучение", type="primary", disabled=running or not selected_paths, key="absa_training_start"):
            _start_training(
                selected_paths, validation_paths, test_paths, int(epochs), int(batch_size),
                float(learning_rate), float(validation_fraction), initial_checkpoint,
            )
            st.rerun()
    with models_tab:
        _render_activation_controls(disabled=running)
    with metrics_tab:
        _render_metrics()
    with comparison_tab:
        _render_marked_epoch_comparison()
    with epochs_tab:
        _render_epoch_management(disabled=running)


@st.fragment(run_every=1.0)
def _render_job_status(job: TrainingJob) -> None:
    progress, message, status, result, error = job.snapshot()
    st.progress(int(progress * 100), text=f"{message} · {int(progress * 100)}%")
    if job.is_running():
        st.info("Обучение выполняется. Изменение его параметров заблокировано.")
        if st.button("Отменить обучение", key="absa_training_cancel"):
            job.cancel_event.set()
            st.warning("Запрошена отмена. Текущий batch завершится, затем обучение безопасно остановится.")
    elif status == "completed" and result is not None:
        st.success(f"Готово. Лучшая эпоха: {result['best_epoch']}.")
    elif status == "cancelled":
        st.warning("Обучение отменено. Завершённые эпохи доступны для выбора или продолжения.")
    elif status == "failed":
        st.error(f"Обучение завершилось ошибкой: {error}")
    if job.consume_completion_refresh():
        st.rerun(scope="app")


def _start_training(
    paths: list[Path], validation_paths: list[Path], test_paths: list[Path], epochs: int,
    batch_size: int, learning_rate: float, validation_fraction: float,
    initial_checkpoint: Path | None,
) -> None:
    run_dir = ARTIFACTS_DIR / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S-%f")
    job = TrainingJob(run_dir=run_dir)

    def worker() -> None:
        try:
            job.update_progress(0.0, "Проверка разметки")
            examples = load_examples(paths)
            validation_examples = load_examples(validation_paths) if validation_paths else None
            test_examples = load_examples(test_paths) if test_paths else None
            result = fine_tune(
                examples, run_dir,
                TrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, validation_fraction=validation_fraction),
                [str(path) for path in paths], validation_examples=validation_examples,
                test_examples=test_examples, validation_files=[str(path) for path in validation_paths],
                test_files=[str(path) for path in test_paths], progress_callback=job.update_progress,
                cancel_check=job.cancel_event.is_set, initial_checkpoint=initial_checkpoint,
            )
            job.finish(result)
        except Exception as error:
            mark_run_status(run_dir, "cancelled" if isinstance(error, TrainingCancelled) else "failed", str(error))
            job.fail(error)

    job.thread = threading.Thread(target=worker, name=f"absa-{run_dir.name}", daemon=True)
    st.session_state[JOB_STATE_KEY] = job
    job.thread.start()


def _render_activation_controls(disabled: bool) -> None:
    if not available_runs():
        return
    st.caption("Сохранённые модели")
    run_dir = _select_run("Запуск", key="absa_active_run", disabled=disabled)
    if run_dir is None:
        return
    manifest = _run_manifest(run_dir)
    status = manifest.get("status", "unknown")
    st.caption(f"Статус запуска: {status}.")
    epochs = sorted(
        int(path.name.removeprefix("epoch-")) for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "config.json").is_file()
    )
    if not epochs:
        return
    active_epoch = manifest.get("active_epoch")
    selected_epoch = st.selectbox(
        "Активная эпоха", options=epochs, index=epochs.index(active_epoch) if active_epoch in epochs else len(epochs) - 1,
        key="absa_active_epoch", disabled=disabled,
    )
    if st.button("Выбрать эпоху", key="absa_activate_epoch", disabled=disabled):
        activate_epoch(run_dir, int(selected_epoch))
        st.success(f"Активирована эпоха {selected_epoch}.")
    completed_epochs = int(manifest.get("last_completed_epoch", 0))
    if status in {"running", "interrupted", "failed", "cancelled"} and 0 < completed_epochs < int(manifest["config"]["epochs"]):
        if st.button("Продолжить с последней эпохи", key="absa_resume_training", disabled=disabled):
            _start_resume(run_dir)
            st.rerun()


def _render_metrics() -> None:
    """Show the saved epoch history without loading model checkpoints."""
    runs = available_runs()
    if not runs:
        st.info("Метрики появятся после сохранения хотя бы одной эпохи.")
        return

    run_dir = _select_run("Запуск для сравнения", key="absa_metrics_run")
    if run_dir is None:
        return
    history_path = run_dir / "history.json"
    if not history_path.is_file():
        st.info("В этом запуске ещё нет завершённых эпох с метриками.")
        return

    history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
    if not history:
        st.info("В этом запуске ещё нет завершённых эпох с метриками.")
        return

    frame = pd.DataFrame(history).sort_values("epoch")
    manifest = _run_manifest(run_dir)
    best_epoch = manifest.get("active_epoch") or json.loads(history_path.read_text(encoding="utf-8")).get("best_epoch")
    if best_epoch:
        validation_title = "внешней validation-выборке" if manifest.get("validation_mode") == "external" else "внутренней валидационной выборке"
        st.success(f"Сейчас выбрана эпоха {best_epoch}. Метрики рассчитаны на {validation_title}.")
    test_metrics_path = run_dir / "test_metrics.json"
    if test_metrics_path.is_file():
        test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        scores = {
            "Joint macro F1": test_metrics["joint"]["macro avg"]["f1-score"],
            "F1 таксономии": test_metrics["taxonomy"]["macro avg"]["f1-score"],
            "F1 тональности": test_metrics["sentiment"]["macro avg"]["f1-score"],
        }
        st.markdown(f"#### Итоговая оценка на test · эпоха {test_metrics['evaluated_epoch']}")
        columns = st.columns(len(scores))
        for column, (label, value) in zip(columns, scores.items()):
            column.metric(label, f"{value:.3f}")

    score_columns = [
        "joint_macro_f1",
        "joint_weighted_f1",
        "taxonomy_macro_f1",
        "sentiment_macro_f1",
    ]
    existing_scores = [column for column in score_columns if column in frame]
    if existing_scores:
        scores_long = frame.melt(id_vars="epoch", value_vars=existing_scores, var_name="Метрика", value_name="Значение")
        labels = {
            "joint_macro_f1": "Macro F1: аспект + тональность",
            "joint_weighted_f1": "Weighted F1: аспект + тональность",
            "taxonomy_macro_f1": "Macro F1: таксономия",
            "sentiment_macro_f1": "Macro F1: тональность",
        }
        scores_long["Метрика"] = scores_long["Метрика"].map(labels)
        st.plotly_chart(
            px.line(scores_long, x="epoch", y="Значение", color="Метрика", markers=True, range_y=[0, 1], title="Качество по эпохам"),
            use_container_width=True,
        )
    loss_columns = [column for column in ("train_loss", "validation_loss") if column in frame]
    if loss_columns:
        losses_long = frame.melt(id_vars="epoch", value_vars=loss_columns, var_name="Потери", value_name="Значение")
        losses_long["Потери"] = losses_long["Потери"].map({"train_loss": "Обучение", "validation_loss": "Валидация"})
        st.plotly_chart(
            px.line(losses_long, x="epoch", y="Значение", color="Потери", markers=True, title="Функция потерь по эпохам"),
            use_container_width=True,
        )
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _epoch_annotations_path(run_dir: Path) -> Path:
    return run_dir / "epoch_annotations.json"


def _read_epoch_annotations(run_dir: Path) -> dict[str, dict[str, object]]:
    path = _epoch_annotations_path(run_dir)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("epochs", {})


def _write_epoch_annotations(run_dir: Path, annotations: dict[str, dict[str, object]]) -> None:
    path = _epoch_annotations_path(run_dir)
    temporary_path = path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps({"epochs": annotations}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary_path, path)


def _all_epoch_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_dir in available_runs():
        manifest = _run_manifest(run_dir)
        annotations = _read_epoch_annotations(run_dir)
        history_path = run_dir / "history.json"
        history_by_epoch = {
            int(item["epoch"]): item
            for item in json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
        } if history_path.is_file() else {}
        for checkpoint in sorted((run_dir / "epochs").glob("epoch-*")):
            if not (checkpoint / "config.json").is_file():
                continue
            epoch = int(checkpoint.name.removeprefix("epoch-"))
            scores = history_by_epoch.get(epoch, {})
            annotation = annotations.get(str(epoch), {})
            rows.append(
                {
                    "Запуск": _run_label(run_dir),
                    "Эпоха": epoch,
                    "Активна": epoch == manifest.get("active_epoch"),
                    "Пометка": bool(annotation.get("marked", False)),
                    "Комментарий": str(annotation.get("note", "")),
                    "Joint macro F1": scores.get("joint_macro_f1"),
                    "F1 таксономии": scores.get("taxonomy_macro_f1"),
                    "F1 тональности": scores.get("sentiment_macro_f1"),
                    "Validation loss": scores.get("validation_loss"),
                }
            )
    return rows


def _render_epoch_management(disabled: bool) -> None:
    """List, annotate, and safely prune completed model checkpoints."""
    rows = _all_epoch_rows()
    if not rows:
        st.info("В сохранённых запусках пока нет завершённых эпох.")
    else:
        st.caption("Пометки сохраняются рядом с запуском и не изменяют веса модели.")
        table = pd.DataFrame(rows).sort_values(["Запуск", "Эпоха"], ascending=[False, True])
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Joint macro F1": st.column_config.NumberColumn(format="%.3f"),
                "F1 таксономии": st.column_config.NumberColumn(format="%.3f"),
                "F1 тональности": st.column_config.NumberColumn(format="%.3f"),
                "Validation loss": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.divider()
    st.subheader("Управление запуском")
    run_dir = _select_run("Запуск", key="absa_manage_run", disabled=disabled)
    if run_dir is None:
        return
    available_epochs = sorted(
        int(path.name.removeprefix("epoch-")) for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "config.json").is_file()
    )
    if not available_epochs:
        st.info("В этом запуске нет сохранённых эпох.")
        if st.button("Удалить запуск", type="primary", disabled=disabled, key="absa_delete_run"):
            _confirm_run_deletion(run_dir)
        return
    epoch = st.selectbox("Эпоха", available_epochs, key="absa_manage_epoch", disabled=disabled)
    annotations = _read_epoch_annotations(run_dir)
    current = annotations.get(str(epoch), {})
    with st.form("absa_epoch_annotation"):
        marked = st.checkbox(
            "Пометить для быстрого сравнения",
            value=bool(current.get("marked", False)),
            key=f"absa_epoch_mark_{run_dir.name}_{epoch}",
            disabled=disabled,
        )
        note = st.text_input(
            "Комментарий к эпохе",
            value=str(current.get("note", "")),
            key=f"absa_epoch_note_{run_dir.name}_{epoch}",
            disabled=disabled,
        )
        save_annotation = st.form_submit_button("Сохранить пометку", disabled=disabled)
    if save_annotation:
        if marked or note.strip():
            annotations[str(epoch)] = {"marked": marked, "note": note.strip()}
        else:
            annotations.pop(str(epoch), None)
        _write_epoch_annotations(run_dir, annotations)
        st.success(f"Пометка эпохи {epoch} сохранена.")
        st.rerun()

    manifest = _run_manifest(run_dir)
    is_active = epoch == manifest.get("active_epoch")
    can_delete = manifest.get("status") == "completed" and not is_active
    publication_action, epoch_action, run_action = st.columns(3)
    with publication_action:
        if is_published(run_dir, epoch):
            if st.button("Снять с публикации", disabled=disabled, key="absa_unpublish_epoch"):
                unpublish_epoch(run_dir, epoch)
                st.rerun()
        elif st.button("Опубликовать для анализа", disabled=disabled, key="absa_publish_epoch"):
            publish_epoch(run_dir, epoch, f"{_run_label(run_dir)} · эпоха {epoch}")
            st.rerun()
    with epoch_action:
        if st.button("Удалить эпоху", type="primary", disabled=disabled or not can_delete, key="absa_delete_epoch"):
            _confirm_epoch_deletion(run_dir, epoch)
    with run_action:
        if st.button("Удалить запуск", type="primary", disabled=disabled, key="absa_delete_run"):
            _confirm_run_deletion(run_dir)
    if is_active:
        st.caption("Активную эпоху удалить нельзя. Сначала выберите другую эпоху во вкладке «Сохранённые модели».")
    elif manifest.get("status") != "completed":
        st.caption("Удаление эпохи незавершённого запуска заблокировано, чтобы сохранить возможность продолжить обучение.")


def _render_marked_epoch_comparison() -> None:
    """Compare only checkpoints explicitly marked by the user."""
    rows = [row for row in _all_epoch_rows() if row["Пометка"]]
    if not rows:
        st.info("Отметьте эпохи во вкладке «Управление эпохами», чтобы добавить их в сравнение.")
        return
    comparison = pd.DataFrame(rows).sort_values(
        "Joint macro F1", ascending=False, na_position="last"
    ).reset_index(drop=True)
    comparison.insert(
        0,
        "Модель",
        comparison["Запуск"] + " · эпоха " + comparison["Эпоха"].astype(str),
    )
    st.caption(f"В сравнении: {len(comparison)} отмеченных эпох.")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Joint macro F1": st.column_config.NumberColumn(format="%.3f"),
            "F1 таксономии": st.column_config.NumberColumn(format="%.3f"),
            "F1 тональности": st.column_config.NumberColumn(format="%.3f"),
            "Validation loss": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    score_columns = ["Joint macro F1", "F1 таксономии", "F1 тональности"]
    chart_data = comparison.melt(
        id_vars="Модель", value_vars=score_columns, var_name="Метрика", value_name="Значение"
    )
    st.plotly_chart(
        px.bar(
            chart_data,
            x="Модель",
            y="Значение",
            color="Метрика",
            barmode="group",
            range_y=[0, 1],
            title="Сравнение отмеченных эпох",
        ),
        use_container_width=True,
    )


@st.dialog("Удалить эпоху?")
def _confirm_epoch_deletion(run_dir: Path, epoch: int) -> None:
    st.warning(f"Будут безвозвратно удалены веса, состояние оптимизатора и метрики эпохи {epoch}.")
    st.caption("Остальные эпохи и сам запуск сохранятся.")
    cancel, confirm = st.columns(2)
    if cancel.button("Отмена", use_container_width=True):
        st.rerun()
    if confirm.button("Удалить безвозвратно", type="primary", use_container_width=True):
        _delete_completed_epoch(run_dir, epoch)
        st.rerun()


@st.dialog("Удалить запуск?")
def _confirm_run_deletion(run_dir: Path) -> None:
    checkpoints = list((run_dir / "epochs").glob("epoch-*/config.json"))
    st.warning(f"Будет безвозвратно удалён весь запуск вместе с {len(checkpoints)} сохранёнными эпохами.")
    st.caption(_run_label(run_dir))
    cancel, confirm = st.columns(2)
    if cancel.button("Отмена", use_container_width=True):
        st.rerun()
    if confirm.button("Удалить безвозвратно", type="primary", use_container_width=True):
        _delete_run(run_dir)
        st.rerun()


def _delete_run(run_dir: Path) -> None:
    """Remove a selected ABSA run after the UI has obtained confirmation."""
    if run_dir.parent != ARTIFACTS_DIR or not (run_dir / "run_manifest.json").is_file():
        raise ValueError("Можно удалить только запуск из каталога артефактов ABSA.")
    unpublish_run(run_dir)
    shutil.rmtree(run_dir)


def _delete_completed_epoch(run_dir: Path, epoch: int) -> None:
    """Delete one non-active completed checkpoint and its history/annotation records."""
    manifest = _run_manifest(run_dir)
    if manifest.get("status") != "completed":
        raise ValueError("Удалять эпохи можно только из завершённого запуска.")
    if epoch == manifest.get("active_epoch"):
        raise ValueError("Нельзя удалить активную эпоху.")
    checkpoint_dir = run_dir / "epochs" / f"epoch-{epoch:03d}"
    if not (checkpoint_dir / "config.json").is_file():
        raise ValueError(f"Эпоха {epoch} не найдена.")
    unpublish_epoch(run_dir, epoch)
    shutil.rmtree(checkpoint_dir)
    history_path = run_dir / "history.json"
    if history_path.is_file():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        payload["epochs"] = [item for item in payload.get("epochs", []) if int(item["epoch"]) != epoch]
        remaining = payload["epochs"]
        payload["best_epoch"] = max(remaining, key=lambda item: item.get("joint_macro_f1", 0.0))["epoch"] if remaining else None
        temporary_path = history_path.with_suffix(".json.tmp")
        temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, history_path)
        csv_path = run_dir / "history.csv"
        if remaining:
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(remaining[0]))
                writer.writeheader()
                writer.writerows(remaining)
        elif csv_path.exists():
            csv_path.unlink()
    annotations = _read_epoch_annotations(run_dir)
    annotations.pop(str(epoch), None)
    _write_epoch_annotations(run_dir, annotations)


def _render_initial_model_picker(disabled: bool) -> Path | None:
    """Keep run and epoch navigation separate as the checkpoint collection grows."""
    choices = ("Исходная RuBERT tiny2", "Сохранённая эпоха")
    source = st.radio(
        "Начать с модели",
        choices,
        horizontal=True,
        key="absa_training_model_source",
        disabled=disabled,
        help="Для сохранённой модели сначала выберите запуск, затем его эпоху.",
    )
    if source == choices[0]:
        return None

    run_dir = _select_run("Запуск исходной модели", key="absa_training_initial_run", disabled=disabled)
    if run_dir is None:
        st.info("Пока нет сохранённых эпох. Для первого запуска используйте исходную RuBERT tiny2.")
        return None
    epochs = sorted(
        int(path.name.removeprefix("epoch-")) for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "config.json").is_file()
    )
    if not epochs:
        st.info("В выбранном запуске нет завершённых эпох.")
        return None
    manifest = _run_manifest(run_dir)
    active_epoch = manifest.get("active_epoch")
    epoch = st.selectbox(
        "Эпоха исходной модели",
        options=epochs,
        index=epochs.index(active_epoch) if active_epoch in epochs else len(epochs) - 1,
        key="absa_training_initial_epoch",
        disabled=disabled,
    )
    return run_dir / "epochs" / f"epoch-{epoch:03d}"


def _start_resume(run_dir: Path) -> None:
    job = TrainingJob(run_dir=run_dir)

    def worker() -> None:
        try:
            result = resume_fine_tune(run_dir, progress_callback=job.update_progress, cancel_check=job.cancel_event.is_set)
            job.finish(result)
        except Exception as error:
            mark_run_status(run_dir, "cancelled" if isinstance(error, TrainingCancelled) else "failed", str(error))
            job.fail(error)

    job.thread = threading.Thread(target=worker, name=f"absa-resume-{run_dir.name}", daemon=True)
    st.session_state[JOB_STATE_KEY] = job
    job.thread.start()

"""Sidebar controls for local aspect-classifier fine-tuning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from absa_training.data import load_examples
from absa_training.trainer import TrainingConfig, activate_epoch, fine_tune, resume_fine_tune
from analysis_helpers.config import MODEL_MAX_TOKENS


DATASETS_DIR = Path("data/absa/train")
ARTIFACTS_DIR = Path("artifacts/absa")


def available_training_datasets() -> list[Path]:
    return sorted(DATASETS_DIR.glob("*_absa_full.csv"))


def available_runs() -> list[Path]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    return sorted(
        (path for path in ARTIFACTS_DIR.iterdir() if path.is_dir() and (path / "run_manifest.json").is_file()),
        reverse=True,
    )


def available_epoch_checkpoints() -> list[Path]:
    return [
        checkpoint
        for run in available_runs()
        for checkpoint in sorted((run / "epochs").glob("epoch-*"), reverse=True)
        if (checkpoint / "config.json").is_file()
    ]


def render_training_sidebar() -> None:
    """Render controls independently of the currently selected analysis source."""
    with st.sidebar:
        st.divider()
        st.subheader("Дообучение ABSA")
        st.caption("Локальное обучение по размеченным аспектам. Во время запуска вкладку не закрывайте.")
        st.caption(f"Контекст модели: {MODEL_MAX_TOKENS} токенов (общая настройка проекта).")
        datasets = available_training_datasets()
        selected_names = st.multiselect(
            "Файлы разметки",
            options=[path.name for path in datasets],
            default=[path.name for path in datasets],
            key="absa_training_datasets",
        )
        epochs = st.number_input("Эпохи", min_value=1, max_value=50, value=4, step=1, key="absa_training_epochs")
        with st.expander("Параметры обучения"):
            batch_size = st.number_input("Размер batch", min_value=1, max_value=128, value=16, step=1, key="absa_training_batch")
            learning_rate = st.number_input(
                "Learning rate", min_value=0.000001, max_value=0.01, value=0.00002,
                step=0.000001, format="%.6f", key="absa_training_lr",
            )
            validation_fraction = st.slider(
                "Доля внутренней валидации", min_value=0.1, max_value=0.4, value=0.2,
                key="absa_training_validation",
            )
        selected_paths = [path for path in datasets if path.name in selected_names]
        checkpoints = available_epoch_checkpoints()
        checkpoint_labels = ["Исходная RuBERT tiny2"] + [f"{path.parent.parent.name} / {path.name}" for path in checkpoints]
        selected_checkpoint_label = st.selectbox(
            "Начать с модели", options=checkpoint_labels,
            help="Без выбора используется исходная RuBERT tiny2. Выбранная эпоха станет начальной точкой нового запуска.",
            key="absa_training_initial_checkpoint",
        )
        initial_checkpoint = None
        if selected_checkpoint_label != checkpoint_labels[0]:
            initial_checkpoint = checkpoints[checkpoint_labels.index(selected_checkpoint_label) - 1]
        if st.button("Запустить дообучение", type="primary", disabled=not selected_paths, key="absa_training_start"):
            _run_training(selected_paths, int(epochs), int(batch_size), float(learning_rate), float(validation_fraction), initial_checkpoint)
        _render_activation_controls()


def _run_training(
    paths: list[Path], epochs: int, batch_size: int, learning_rate: float, validation_fraction: float,
    initial_checkpoint: Path | None,
) -> None:
    progress = st.progress(0, text="Проверка разметки")
    started_at = datetime.now()
    run_dir = ARTIFACTS_DIR / started_at.strftime("run_%Y-%m-%d_%H-%M-%S")

    def on_progress(value: float, message: str) -> None:
        progress.progress(min(int(value * 100), 100), text=f"{message} · {int(value * 100)}%")

    try:
        examples = load_examples(paths)
        result = fine_tune(
            examples,
            run_dir,
            TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_fraction=validation_fraction,
            ),
            [str(path) for path in paths],
            progress_callback=on_progress,
            initial_checkpoint=initial_checkpoint,
        )
    except Exception as error:
        progress.empty()
        st.error(f"Дообучение не завершено: {error}")
        return
    elapsed = datetime.now() - started_at
    progress.progress(100, text="Дообучение завершено")
    st.success(f"Готово за {elapsed.seconds // 60} мин {elapsed.seconds % 60} сек. Лучшая эпоха: {result['best_epoch']}.")


def _render_activation_controls() -> None:
    runs = available_runs()
    if not runs:
        return
    st.caption("Сохранённые модели")
    selected_name = st.selectbox("Запуск", options=[path.name for path in runs], key="absa_active_run")
    run_dir = next(path for path in runs if path.name == selected_name)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("status", "unknown")
    st.caption(f"Статус запуска: {status}.")
    epochs = sorted(
        int(path.name.removeprefix("epoch-"))
        for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "config.json").is_file()
    )
    if not epochs:
        return
    active_epoch = manifest.get("active_epoch")
    selected_epoch = st.selectbox(
        "Активная эпоха", options=epochs,
        index=epochs.index(active_epoch) if active_epoch in epochs else len(epochs) - 1,
        key="absa_active_epoch",
    )
    if st.button("Выбрать эпоху", key="absa_activate_epoch"):
        activate_epoch(run_dir, int(selected_epoch))
        st.success(f"Активирована эпоха {selected_epoch}.")
    completed_epochs = int(manifest.get("last_completed_epoch", 0))
    if status in {"running", "interrupted", "failed"} and 0 < completed_epochs < int(manifest["config"]["epochs"]):
        if st.button("Продолжить с последней эпохи", key="absa_resume_training"):
            _resume_training(run_dir)


def _resume_training(run_dir: Path) -> None:
    progress = st.progress(0, text="Подготовка продолжения обучения")

    def on_progress(value: float, message: str) -> None:
        progress.progress(min(int(value * 100), 100), text=f"{message} · {int(value * 100)}%")

    try:
        result = resume_fine_tune(run_dir, progress_callback=on_progress)
    except Exception as error:
        progress.empty()
        st.error(f"Не удалось продолжить обучение: {error}")
        return
    progress.progress(100, text="Дообучение завершено")
    st.success(f"Продолжение завершено. Лучшая эпоха: {result['best_epoch']}.")

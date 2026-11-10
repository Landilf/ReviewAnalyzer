"""Reproducible visual comparison of the rule baseline and an ABSA checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from absa_training.data import AspectExample, load_examples
from absa_training.legacy_baseline import legacy_taxonomy
from absa_training.metrics import build_absa_metrics, primary_scores
from analysis_helpers.config import MODEL_MAX_TOKENS
from analysis_helpers.pipeline import normalize_sentiment
from analysis_helpers.sentiment import sentiment_with_transformer


ARTIFACTS_DIR = Path("artifacts/absa")
TEST_DIR = Path("data/absa/test")
UNCLASSIFIED = "не определено"


def _available_runs() -> list[Path]:
    if not ARTIFACTS_DIR.is_dir():
        return []
    return sorted(
        (
            path for path in ARTIFACTS_DIR.iterdir()
            if (path / "run_manifest.json").is_file()
            and any((path / "epochs").glob("epoch-*/config.json"))
        ),
        reverse=True,
    )


def _run_label(run_dir: Path) -> str:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    return f"{run_dir.name.removeprefix('run_').replace('_', ' ')} · {manifest.get('status', 'unknown')}"


def _checkpoint_options(run_dir: Path) -> list[int]:
    return sorted(
        int(path.name.removeprefix("epoch-"))
        for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "config.json").is_file()
    )


def _predict_absa(examples: list[AspectExample], checkpoint: Path) -> list[str]:
    """Predict labels in batches without changing the selected checkpoint."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions: list[str] = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            encoded = tokenizer(
                [item.marked_text() for item in batch], truncation=True,
                max_length=MODEL_MAX_TOKENS, padding=True, return_tensors="pt",
            )
            output = model(**{name: value.to(device) for name, value in encoded.items()})
            predictions.extend(model.config.id2label[int(index)] for index in output.logits.argmax(dim=-1).cpu().tolist())
    return predictions


def _predict_review_sentiments(examples: list[AspectExample]) -> dict[tuple[str, str], str]:
    """Run the legacy review-level classifier once per review, not per aspect."""
    unique_reviews = {(item.source_file, item.review_id): item.text for item in examples}
    classifier = sentiment_with_transformer()
    keys = list(unique_reviews)
    raw = classifier([unique_reviews[key] for key in keys], truncation=True, max_length=MODEL_MAX_TOKENS)
    return {key: normalize_sentiment(prediction["label"]) for key, prediction in zip(keys, raw)}


def _run_comparison(examples: list[AspectExample], checkpoint: Path) -> dict:
    true_labels = [item.label for item in examples]
    checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    taxonomy_labels = sorted({
        label.rsplit("::", maxsplit=1)[0]
        for label in checkpoint_config.get("id2label", {}).values()
    }) or sorted({item.taxonomy for item in examples})
    absa_labels = _predict_absa(examples, checkpoint)
    review_sentiments = _predict_review_sentiments(examples)
    legacy_taxonomies = [legacy_taxonomy(item.aspect) for item in examples]
    legacy_labels = [
        f"{taxonomy or UNCLASSIFIED}::{review_sentiments[(item.source_file, item.review_id)]}"
        for item, taxonomy in zip(examples, legacy_taxonomies)
    ]
    return {
        "absa_metrics": build_absa_metrics(true_labels, absa_labels, taxonomy_labels),
        "legacy_metrics": build_absa_metrics(true_labels, legacy_labels, taxonomy_labels),
        "coverage": sum(value is not None for value in legacy_taxonomies) / len(examples),
        "rows": pd.DataFrame({
            "Текст": [item.text for item in examples],
            "Аспект": [item.aspect for item in examples],
            "Истинная таксономия": [item.taxonomy for item in examples],
            "Истинная тональность": [item.sentiment for item in examples],
            "Правила: таксономия": [value or UNCLASSIFIED for value in legacy_taxonomies],
            "Правила: тональность": [label.rsplit("::", maxsplit=1)[1] for label in legacy_labels],
            "ABSA: таксономия": [label.rsplit("::", maxsplit=1)[0] for label in absa_labels],
            "ABSA: тональность": [label.rsplit("::", maxsplit=1)[1] for label in absa_labels],
        }),
    }


def _metrics_frame(comparison: dict) -> pd.DataFrame:
    legacy = primary_scores(comparison["legacy_metrics"])
    absa = primary_scores(comparison["absa_metrics"])
    return pd.DataFrame([
        {"Метод": "Правиловый baseline", "Покрытие таксономии": comparison["coverage"], **legacy},
        {"Метод": "Дообученная ABSA-модель", "Покрытие таксономии": 1.0, **absa},
    ]).rename(columns={
        "joint_macro_f1": "Joint macro F1",
        "joint_weighted_f1": "Joint weighted F1",
        "taxonomy_macro_f1": "Taxonomy macro F1",
        "sentiment_macro_f1": "Sentiment macro F1",
    })


def _taxonomy_frame(comparison: dict) -> pd.DataFrame:
    records: list[dict] = []
    for taxonomy, result in comparison["absa_metrics"]["per_taxonomy"].items():
        legacy = comparison["legacy_metrics"]["per_taxonomy"].get(taxonomy, {})
        records.append({
            "Таксономия": taxonomy,
            "Аспектов": result["support"],
            "Правила: joint accuracy": legacy.get("joint_accuracy", 0.0),
            "ABSA: joint accuracy": result["joint_accuracy"],
        })
    return pd.DataFrame(records).sort_values("Аспектов", ascending=False)


def render_experiment_page() -> None:
    """Render controls and article-ready figures for the fixed test split."""
    st.title("Экспериментальное сравнение")
    st.caption("Одинаковые размеченные аспекты, единая таксономия и независимая test-выборка.")
    test_paths = sorted(TEST_DIR.glob("*.csv"))
    runs = _available_runs()
    if not test_paths:
        st.error("Не найдены CSV-файлы в data/absa/test.")
        return
    if not runs:
        st.error("Нет сохранённых запусков ABSA для сравнения.")
        return

    left, right = st.columns(2)
    with left:
        run_dir = st.selectbox("Запуск ABSA", runs, format_func=_run_label, key="experiment_run")
    with right:
        epochs = _checkpoint_options(run_dir)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        default_epoch = manifest.get("active_epoch")
        epoch = st.selectbox("Эпоха", epochs, index=epochs.index(default_epoch) if default_epoch in epochs else len(epochs) - 1, key="experiment_epoch")

    examples = load_examples(test_paths)
    st.info(
        f"Test: {len(examples):,} аспектов из {len({item.group_id for item in examples}):,} отзывов. "
        "Границы аспектов фиксированы для обоих методов; сравниваются только таксономия и тональность."
    )
    st.caption(
        "Правиловый baseline: термин сопоставляется со словарём старого метода, "
        "а тональность всего отзыва переносится на каждый его аспект. "
        "Пропуск категории учитывается как ошибка и отдельно показывается в покрытии."
    )
    cache_key = f"experiment_result_{run_dir.name}_{epoch}_{','.join(path.name for path in test_paths)}"
    if st.button("Сформировать сравнение", type="primary", key="experiment_run_comparison"):
        with st.spinner("Запускаю оба метода на фиксированной test-выборке…"):
            st.session_state[cache_key] = _run_comparison(examples, run_dir / "epochs" / f"epoch-{epoch:03d}")
    comparison = st.session_state.get(cache_key)
    if comparison is None:
        st.info("Нажмите «Сформировать сравнение», чтобы получить таблицы и графики для статьи.")
        return

    metrics = _metrics_frame(comparison)
    st.subheader("Основное сравнение")
    st.dataframe(metrics, use_container_width=True, hide_index=True, column_config={
        column: st.column_config.NumberColumn(format="%.3f")
        for column in metrics.columns if column != "Метод"
    })
    chart = metrics.melt(id_vars="Метод", value_vars=["Taxonomy macro F1", "Sentiment macro F1", "Joint macro F1"], var_name="Метрика", value_name="F1")
    st.plotly_chart(px.bar(chart, x="Метрика", y="F1", color="Метод", barmode="group", range_y=[0, 1], title="Macro F1 на независимой test-выборке"), use_container_width=True)

    taxonomy = _taxonomy_frame(comparison)
    st.subheader("Качество по таксономиям")
    shown = taxonomy[taxonomy["Аспектов"] >= 10].copy()
    long = shown.melt(id_vars=["Таксономия", "Аспектов"], value_vars=["Правила: joint accuracy", "ABSA: joint accuracy"], var_name="Метод", value_name="Joint accuracy")
    st.plotly_chart(px.bar(long, x="Таксономия", y="Joint accuracy", color="Метод", barmode="group", range_y=[0, 1], title="Совместная точность по частым аспектам"), use_container_width=True)
    st.dataframe(taxonomy, use_container_width=True, hide_index=True, column_config={
        "Правила: joint accuracy": st.column_config.NumberColumn(format="%.3f"),
        "ABSA: joint accuracy": st.column_config.NumberColumn(format="%.3f"),
    })

    st.subheader("Примеры расхождений")
    rows = comparison["rows"]
    errors = rows[(rows["Правила: таксономия"] != rows["Истинная таксономия"]) | (rows["ABSA: таксономия"] != rows["Истинная таксономия"])]
    st.dataframe(errors.head(30), use_container_width=True, hide_index=True)

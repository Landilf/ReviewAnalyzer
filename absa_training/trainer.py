"""Epoch checkpointing and fine-tuning for taxonomy-aware ABSA classification."""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np

from analysis_helpers.config import MODEL_MAX_TOKENS
from absa_training.data import AspectExample, load_examples, split_by_review
from absa_training.metrics import build_absa_metrics, primary_scores


ProgressCallback = Callable[[float, str], None] | None


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "cointegrated/rubert-tiny2"
    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 2e-5
    validation_fraction: float = 0.2
    random_state: int = 42


def _write_json(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, path)


def _write_history(run_dir: Path, history: list[dict], best_epoch: int) -> None:
    if not history:
        return
    history_path = run_dir / "history.csv"
    temporary_path = history_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary_path, history_path)
    _write_json(run_dir / "history.json", {"epochs": history, "best_epoch": best_epoch})


def _checkpoint_epochs(run_dir: Path) -> list[int]:
    return sorted(
        int(path.name.removeprefix("epoch-"))
        for path in (run_dir / "epochs").glob("epoch-*")
        if (path / "training_state.pt").is_file() and (path / "metrics.json").is_file()
    )


def _restore_history(run_dir: Path, last_epoch: int) -> list[dict]:
    """Restore history from checkpoints if an interruption happened between writes."""
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", []) if history_path.is_file() else []
    existing_epochs = {item["epoch"] for item in history}
    for epoch in _checkpoint_epochs(run_dir):
        if epoch > last_epoch or epoch in existing_epochs:
            continue
        payload = json.loads((run_dir / "epochs" / f"epoch-{epoch:03d}" / "metrics.json").read_text(encoding="utf-8"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": payload.get("train_loss", float("nan")),
                "validation_loss": payload["validation_loss"],
                **primary_scores(payload),
            }
        )
    return sorted(history, key=lambda item: item["epoch"])


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _Dataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        import torch

        item = {name: torch.tensor(values[index]) for name, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def _evaluate(model, loader, device, labels: list[str]) -> tuple[dict, float]:
    import torch

    model.eval()
    predictions: list[str] = []
    actual: list[str] = []
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            output = model(**batch)
            losses.append(float(output.loss.item()))
            predicted_ids = output.logits.argmax(dim=-1).detach().cpu().tolist()
            actual_ids = batch["labels"].detach().cpu().tolist()
            predictions.extend(labels[item] for item in predicted_ids)
            actual.extend(labels[item] for item in actual_ids)
    taxonomies = sorted({label.split("::", maxsplit=1)[0] for label in labels})
    return build_absa_metrics(actual, predictions, taxonomies), float(np.mean(losses))


def _save_epoch(
    model,
    tokenizer,
    optimizer,
    scheduler,
    run_dir: Path,
    epoch: int,
    metrics: dict,
    validation_loss: float,
    train_loss: float,
) -> Path:
    import torch

    checkpoint_dir = run_dir / "epochs" / f"epoch-{epoch:03d}"
    temporary_dir = run_dir / "epochs" / f".epoch-{epoch:03d}.incomplete"
    if checkpoint_dir.exists():
        raise ValueError(f"Эпоха {epoch} уже сохранена: {checkpoint_dir}")
    if temporary_dir.exists():
        raise RuntimeError(f"Найдена незавершённая запись эпохи: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    model.save_pretrained(temporary_dir)
    tokenizer.save_pretrained(temporary_dir)
    torch.save(
        {"epoch": epoch, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        temporary_dir / "training_state.pt",
    )
    _write_json(temporary_dir / "metrics.json", {"train_loss": train_loss, "validation_loss": validation_loss, **metrics})
    os.replace(temporary_dir, checkpoint_dir)
    return checkpoint_dir


def activate_epoch(run_dir: Path, epoch: int) -> Path:
    """Select an existing epoch for inference without deleting newer checkpoints."""
    checkpoint_dir = run_dir / "epochs" / f"epoch-{epoch:03d}"
    if not (checkpoint_dir / "config.json").is_file():
        raise ValueError(f"Контрольная точка эпохи {epoch} не найдена: {checkpoint_dir}")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Не найден манифест запуска: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["active_epoch"] = epoch
    manifest["active_checkpoint"] = str(checkpoint_dir.relative_to(run_dir))
    _write_json(manifest_path, manifest)
    return checkpoint_dir


def fine_tune(
    examples: list[AspectExample],
    run_dir: Path,
    config: TrainingConfig,
    source_files: list[str],
    progress_callback: ProgressCallback = None,
    resume_epoch: int | None = None,
    initial_checkpoint: Path | None = None,
) -> dict:
    """Train one classifier for ``taxonomy::sentiment`` labels and save every epoch."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs и batch_size должны иметь допустимые положительные значения.")
    if resume_epoch is not None and initial_checkpoint is not None:
        raise ValueError("Нельзя одновременно продолжать запуск и выбирать другую стартовую эпоху.")
    _seed_everything(config.random_state)
    train_examples, validation_examples = split_by_review(examples, config.validation_fraction, config.random_state)
    labels = sorted({example.label for example in examples})
    label_to_id = {label: index for index, label in enumerate(labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = run_dir / "run_manifest.json"
    is_resuming = resume_epoch is not None
    if is_resuming:
        if not manifest_path.is_file():
            raise ValueError("Для продолжения нужен существующий run_manifest.json.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config") != asdict(config) or manifest.get("labels") != labels:
            raise ValueError("Параметры, набор меток или разбиение не совпадают с исходным запуском.")
        checkpoint_dir = run_dir / "epochs" / f"epoch-{resume_epoch:03d}"
        if not (checkpoint_dir / "training_state.pt").is_file():
            raise ValueError(f"Нельзя продолжить: не найдена завершённая эпоха {resume_epoch}.")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        start_epoch = resume_epoch + 1
        history = _restore_history(run_dir, resume_epoch)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "epochs").mkdir()
        manifest = {
            "created_at": datetime.now(UTC).isoformat(),
            "source_files": source_files,
            "config": asdict(config),
            "device": str(device),
            "labels": labels,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "active_epoch": None,
            "active_checkpoint": None,
            "last_completed_epoch": 0,
            "status": "running",
        }
        _write_json(manifest_path, manifest)
        if initial_checkpoint is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ASPECT]", "[/ASPECT]"]})
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=len(labels),
                id2label={index: label for index, label in enumerate(labels)},
                label2id=label_to_id,
            )
            model.resize_token_embeddings(len(tokenizer))
            initialization = config.model_name
        else:
            if not (initial_checkpoint / "config.json").is_file():
                raise ValueError(f"Не найдена выбранная стартовая эпоха: {initial_checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(initial_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(initial_checkpoint)
            if model.config.label2id != label_to_id:
                raise ValueError("Выбранная эпоха обучена на другой таксономии или другом наборе тональностей.")
            initialization = str(initial_checkpoint)
        start_epoch = 1
        history = []
        manifest["initialization"] = initialization
        _write_json(manifest_path, manifest)
    model.to(device)

    def build_loader(items: list[AspectExample], shuffle: bool) -> DataLoader:
        encodings = tokenizer(
            [item.marked_text() for item in items],
            truncation=True,
            max_length=MODEL_MAX_TOKENS,
            padding=True,
        )
        dataset = _Dataset(encodings, [label_to_id[item.label] for item in items])
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

    train_loader = build_loader(train_examples, shuffle=True)
    validation_loader = build_loader(validation_examples, shuffle=False)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * config.epochs,
    )
    if is_resuming:
        checkpoint_state = torch.load(run_dir / "epochs" / f"epoch-{resume_epoch:03d}" / "training_state.pt", map_location=device)
        optimizer.load_state_dict(checkpoint_state["optimizer"])
        scheduler.load_state_dict(checkpoint_state["scheduler"])
    manifest["status"] = "running"
    manifest.pop("error", None)
    _write_json(manifest_path, manifest)
    if progress_callback is not None:
        progress_callback(0.0, "Подготовка модели и набора данных")
    best_epoch = max((item["epoch"] for item in history), default=0)
    best_score = max((item["joint_macro_f1"] for item in history), default=float("-inf"))
    try:
      for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(output.loss.item()))
            if progress_callback is not None:
                completed = ((epoch - 1) + batch_index / len(train_loader)) / config.epochs
                progress_callback(completed * 0.9, f"Эпоха {epoch}/{config.epochs}: batch {batch_index}/{len(train_loader)}")
        metrics, validation_loss = _evaluate(model, validation_loader, device, labels)
        scores = primary_scores(metrics)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": validation_loss,
            **scores,
        }
        history.append(epoch_record)
        _save_epoch(model, tokenizer, optimizer, scheduler, run_dir, epoch, metrics, validation_loss, epoch_record["train_loss"])
        if progress_callback is not None:
            progress_callback(epoch / config.epochs, f"Эпоха {epoch}/{config.epochs} сохранена")
        if scores["joint_macro_f1"] > best_score:
            best_epoch = epoch
            best_score = scores["joint_macro_f1"]
        _write_history(run_dir, history, best_epoch)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["last_completed_epoch"] = epoch
        _write_json(manifest_path, manifest)
    except KeyboardInterrupt:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "interrupted"
        manifest["last_completed_epoch"] = max((item["epoch"] for item in history), default=0)
        _write_json(manifest_path, manifest)
        raise
    except Exception as error:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "failed"
        manifest["last_completed_epoch"] = max((item["epoch"] for item in history), default=0)
        manifest["error"] = f"{type(error).__name__}: {error}"
        _write_json(manifest_path, manifest)
        raise
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "completed"
    _write_json(manifest_path, manifest)
    activate_epoch(run_dir, best_epoch)
    if progress_callback is not None:
        progress_callback(1.0, f"Обучение завершено; выбрана эпоха {best_epoch}")
    return {"run_dir": str(run_dir), "best_epoch": best_epoch, "history": history}


def resume_fine_tune(run_dir: Path, progress_callback: ProgressCallback = None) -> dict:
    """Continue a failed or interrupted run from its last complete epoch."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Не найден манифест запуска: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    last_epoch = max([int(manifest.get("last_completed_epoch", 0)), *_checkpoint_epochs(run_dir)], default=0)
    total_epochs = int(manifest["config"]["epochs"])
    if last_epoch < 1:
        raise ValueError("Нет завершённой эпохи, с которой можно продолжить обучение.")
    if last_epoch >= total_epochs:
        raise ValueError("Все запланированные эпохи уже завершены.")
    source_files = [Path(path) for path in manifest["source_files"]]
    examples = load_examples(source_files)
    return fine_tune(
        examples, run_dir, TrainingConfig(**manifest["config"]), manifest["source_files"],
        progress_callback=progress_callback, resume_epoch=last_epoch,
    )

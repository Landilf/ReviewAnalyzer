"""Registry of ABSA checkpoints explicitly published for future analysis use."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts/absa")
REGISTRY_PATH = ARTIFACTS_DIR / "published_models.json"


def _write_registry(payload: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = REGISTRY_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_path, REGISTRY_PATH)


def list_published_models() -> list[dict]:
    """Return only registry entries whose checkpoint still exists."""
    if not REGISTRY_PATH.is_file():
        return []
    models = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("models", [])
    return [
        model for model in models
        if (ARTIFACTS_DIR / model.get("checkpoint", "") / "config.json").is_file()
    ]


def is_published(run_dir: Path, epoch: int) -> bool:
    checkpoint = str((run_dir / "epochs" / f"epoch-{epoch:03d}").relative_to(ARTIFACTS_DIR))
    return any(model.get("checkpoint") == checkpoint for model in list_published_models())


def publish_epoch(run_dir: Path, epoch: int, display_name: str) -> None:
    """Publish one completed checkpoint without changing its run manifest."""
    checkpoint_dir = run_dir / "epochs" / f"epoch-{epoch:03d}"
    if not (checkpoint_dir / "config.json").is_file():
        raise ValueError(f"Эпоха {epoch} не найдена: {checkpoint_dir}")
    checkpoint = str(checkpoint_dir.relative_to(ARTIFACTS_DIR))
    models = list_published_models()
    if any(model.get("checkpoint") == checkpoint for model in models):
        return
    models.append(
        {
            "checkpoint": checkpoint,
            "display_name": display_name,
            "published_at": datetime.now(UTC).isoformat(),
        }
    )
    _write_registry({"models": models})


def unpublish_epoch(run_dir: Path, epoch: int) -> None:
    checkpoint = str((run_dir / "epochs" / f"epoch-{epoch:03d}").relative_to(ARTIFACTS_DIR))
    _write_registry({"models": [model for model in list_published_models() if model.get("checkpoint") != checkpoint]})


def unpublish_run(run_dir: Path) -> None:
    run_prefix = f"{run_dir.name}/"
    _write_registry(
        {"models": [model for model in list_published_models() if not str(model.get("checkpoint", "")).startswith(run_prefix)]}
    )

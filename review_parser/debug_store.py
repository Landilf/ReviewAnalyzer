from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug_exports" / "parsed_reviews"


def save_parsed_reviews(reviews: pd.DataFrame, source: str, label: str | None = None, keep_last: int = 20) -> dict[str, Path]:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_source = _slugify(source or "unknown-source")
    safe_label = _slugify(label or "parsed")
    base_name = f"{timestamp}_{safe_label}_{safe_source}"

    csv_path = DEBUG_DIR / f"{base_name}.csv"
    meta_path = DEBUG_DIR / f"{base_name}.json"

    reviews.to_csv(csv_path, index=False, encoding="utf-8-sig")
    metadata = {
        "source": source,
        "label": label,
        "created_at": timestamp,
        "rows": int(len(reviews)),
        "columns": list(reviews.columns),
        "csv_path": csv_path.name,
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    _prune_old_snapshots(keep_last=keep_last)
    return {"csv": csv_path, "json": meta_path}


def _prune_old_snapshots(keep_last: int = 20) -> None:
    csv_files = sorted(
        DEBUG_DIR.glob("*.csv"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for csv_file in csv_files[:-keep_last]:
        meta_file = csv_file.with_suffix(".json")
        try:
            csv_file.unlink()
        except OSError:
            pass
        try:
            if meta_file.exists():
                meta_file.unlink()
        except OSError:
            pass


def _slugify(value: str) -> str:
    normalized = value.strip().lower().replace(" ", "-")
    normalized = re.sub(r"[^a-z0-9а-яё._-]+", "-", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-_.")
    return normalized or "unknown"

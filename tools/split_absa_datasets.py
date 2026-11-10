"""Create fixed train/validation/test ABSA splits without leaking whole reviews."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION = PROJECT_ROOT / "data/absa/validation/validation_absa.csv"
DEFAULT_TEST = PROJECT_ROOT / "data/absa/test/test_absa.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/absa/split_manifest.json"
SPLITS = ("train", "validation", "test")


def _stable_order(source_name: str, review_ids: list[str], seed: int) -> list[str]:
    def digest(review_id: str) -> bytes:
        value = f"{seed}:{source_name}:{review_id}".encode("utf-8")
        return hashlib.sha256(value).digest()

    return sorted(review_ids, key=digest)


def _split_review_ids(source_name: str, review_ids: list[str], seed: int) -> dict[str, str]:
    ordered = _stable_order(source_name, review_ids, seed)
    count = len(ordered)
    validation_count = max(1, round(count * 0.1))
    test_count = max(1, round(count * 0.1))
    if validation_count + test_count >= count:
        raise ValueError(f"{source_name}: недостаточно отзывов для разбиения 80/10/10.")
    assignment = {review_id: "test" for review_id in ordered[:test_count]}
    assignment.update({review_id: "validation" for review_id in ordered[test_count:test_count + validation_count]})
    assignment.update({review_id: "train" for review_id in ordered[test_count + validation_count:]})
    return assignment


def _read_source(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        if not fieldnames or "review_id" not in fieldnames:
            raise ValueError(f"{path}: не найден обязательный столбец review_id.")
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            groups[row["review_id"]].append(row)
    return fieldnames, groups


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return temporary_path


def split_datasets(
    sources: list[Path], validation_path: Path, test_path: Path, manifest_path: Path, seed: int, force: bool,
) -> dict:
    if not sources:
        raise ValueError("Не переданы исходные CSV для разбиения.")
    for output in (validation_path, test_path, manifest_path):
        if output.exists() and not force:
            raise FileExistsError(f"{output} уже существует. Используйте --force для перезаписи.")
        output.parent.mkdir(parents=True, exist_ok=True)

    combined: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    manifest_sources: list[dict] = []
    expected_fields: list[str] | None = None
    train_rewrites: list[tuple[Path, list[str], list[dict[str, str]]]] = []

    for source in sources:
        fieldnames, groups = _read_source(source)
        if expected_fields is None:
            expected_fields = fieldnames
        elif fieldnames != expected_fields:
            raise ValueError(f"{source}: набор столбцов отличается от {sources[0]}.")
        assignment = _split_review_ids(source.name, list(groups), seed)
        train_rows: list[dict[str, str]] = []
        source_stats: dict[str, dict[str, int]] = {split: {"reviews": 0, "rows": 0} for split in SPLITS}
        for review_id, rows in groups.items():
            split = assignment[review_id]
            source_stats[split]["reviews"] += 1
            source_stats[split]["rows"] += len(rows)
            if split == "train":
                train_rows.extend(rows)
            else:
                combined[split].extend({"source_dataset": source.name, **row} for row in rows)
        train_rewrites.append((source, fieldnames, train_rows))
        manifest_sources.append({"source": str(source), **source_stats})

    assert expected_fields is not None
    validation_fields = ["source_dataset", *expected_fields]
    temporary_files = [
        _write_csv(validation_path, validation_fields, combined["validation"]),
        _write_csv(test_path, validation_fields, combined["test"]),
    ]
    temporary_files.extend(_write_csv(path, fields, rows) for path, fields, rows in train_rewrites)
    manifest = {
        "seed": seed,
        "strategy": "Deterministic 80/10/10 split by source_dataset + review_id.",
        "sources": manifest_sources,
        "totals": {
            split: {
                "rows": len(combined[split]) if split != "train" else sum(len(rows) for _, _, rows in train_rewrites),
                "reviews": sum(item[split]["reviews"] for item in manifest_sources),
            }
            for split in SPLITS
        },
    }
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_files.append(temporary_manifest)

    for temporary_path in temporary_files:
        os.replace(temporary_path, temporary_path.with_name(temporary_path.name.removeprefix("." ).removesuffix(".tmp")))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="CSV в data/absa/train, которые нужно разделить")
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = split_datasets(args.sources, args.validation, args.test, args.manifest, args.seed, args.force)
    for split, values in manifest["totals"].items():
        print(f"{split}: {values['reviews']} отзывов, {values['rows']} аспектных строк")


if __name__ == "__main__":
    main()

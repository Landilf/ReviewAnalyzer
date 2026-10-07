"""Loading and validation of aspect-level annotation CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


REQUIRED_COLUMNS = {
    "review_id",
    "text",
    "aspect",
    "aspect_normalized",
    "aspect_start",
    "aspect_end",
    "aspect_sentiment",
    "annotation_status",
}
SENTIMENTS = {"negative", "neutral", "positive"}
REJECTED_STATUSES = {"rejected", "reject", "deleted"}


@dataclass(frozen=True)
class AspectExample:
    source_file: str
    review_id: str
    text: str
    aspect: str
    taxonomy: str
    sentiment: str
    start: int
    end: int
    status: str

    @property
    def group_id(self) -> str:
        return f"{self.source_file}:{self.review_id}"

    @property
    def label(self) -> str:
        return f"{self.taxonomy}::{self.sentiment}"

    def marked_text(self) -> str:
        return f"{self.text[:self.start]} [ASPECT] {self.text[self.start:self.end]} [/ASPECT] {self.text[self.end:]}"


def load_examples(paths: list[Path], allowed_statuses: set[str] | None = None) -> list[AspectExample]:
    """Load accepted annotation candidates and fail fast on invalid spans.

    By default every status except an explicit rejection is included. This keeps
    the loader usable with externally reviewed datasets that use their own
    status vocabulary, while still allowing a strict status allow-list.
    """
    examples: list[AspectExample] = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise ValueError(f"{path}: отсутствуют обязательные столбцы: {', '.join(sorted(missing))}.")
            for row_number, row in enumerate(reader, start=2):
                status = row["annotation_status"].strip().lower()
                if status in REJECTED_STATUSES or (allowed_statuses is not None and status not in allowed_statuses):
                    continue
                text = row["text"]
                try:
                    start = int(row["aspect_start"])
                    end = int(row["aspect_end"])
                except ValueError as error:
                    raise ValueError(f"{path}:{row_number}: координаты аспекта должны быть целыми числами.") from error
                aspect = row["aspect"]
                if not (0 <= start < end <= len(text)) or text[start:end] != aspect:
                    raise ValueError(f"{path}:{row_number}: aspect_start/aspect_end не соответствуют тексту аспекта.")
                sentiment = row["aspect_sentiment"].strip().lower()
                if sentiment not in SENTIMENTS:
                    raise ValueError(f"{path}:{row_number}: неизвестная тональность аспекта: {sentiment!r}.")
                taxonomy = row["aspect_normalized"].strip()
                if not taxonomy:
                    raise ValueError(f"{path}:{row_number}: не заполнена таксономия аспекта.")
                key = (path.name, row["review_id"], start, end)
                if key in seen:
                    continue
                seen.add(key)
                examples.append(
                    AspectExample(
                        source_file=path.name,
                        review_id=row["review_id"],
                        text=text,
                        aspect=aspect,
                        taxonomy=taxonomy,
                        sentiment=sentiment,
                        start=start,
                        end=end,
                        status=status,
                    )
                )
    if not examples:
        raise ValueError("После фильтрации не осталось строк для обучения.")
    return examples


def split_by_review(
    examples: list[AspectExample], validation_fraction: float, random_state: int,
) -> tuple[list[AspectExample], list[AspectExample]]:
    """Split by review identity, preventing aspect spans from the same review leaking."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction должен быть между 0 и 1.")
    group_ids = [example.group_id for example in examples]
    if len(set(group_ids)) < 2:
        raise ValueError("Для разделения нужны как минимум два разных отзыва.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=random_state)
    train_indices, validation_indices = next(splitter.split(examples, groups=group_ids))
    return (
        [examples[index] for index in train_indices],
        [examples[index] for index in validation_indices],
    )

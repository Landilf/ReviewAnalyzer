"""Create a review-level ABSA pre-annotation file from Otzovik pros/cons blocks.

The result is deliberately marked ``needs_review``: it is an annotation queue,
not ground truth.  Only aspect mentions in the author's explicit
"Достоинства" / "Недостатки" blocks are emitted, so that each proposed
polarity has an inspectable textual basis.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ASPECT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("аккумулятор", re.compile(r"\b(?:акб|аккумулятор\w*|батаре\w*|автономност\w*|заряд(?:а|у|ом|е)?)\b", re.I)),
    ("камера", re.compile(r"\b(?:камер\w*|фотограф\w*|фото)\b", re.I)),
    ("экран", re.compile(r"\b(?:экран\w*|диспле\w*|яркост\w*|цветопередач\w*|разрешени\w*)", re.I)),
    ("память", re.compile(r"\b(?:памят\w*|пзу|озу|хранилищ\w*)", re.I)),
    ("производительность", re.compile(r"\b(?:производительност\w*|скорост\w*|быстродейств\w*)", re.I)),
    ("процессор", re.compile(r"\b(?:процессор\w*|чип\w*)", re.I)),
    ("цена", re.compile(r"\b(?:цен\w*|стоимост\w*)", re.I)),
    ("дизайн", re.compile(r"\b(?:дизайн\w*|внешн\w* вид\w*|вид)\b", re.I)),
    ("корпус", re.compile(r"\b(?:корпус\w*|крышк\w*|пластик\w*|материал\w*|сборк\w*)", re.I)),
    ("качество", re.compile(r"\bкачеств(?:о|а|у|ом|е)?\b", re.I)),
    ("звук", re.compile(r"\b(?:звук\w*|динамик\w*|микрофон\w*)", re.I)),
    ("зарядка", re.compile(r"\b(?:разъ[её]м\w*|гнезд\w*)\s+(?:для\s+)?зарядк\w*\b|\bзарядк\w*", re.I)),
    ("комплектация", re.compile(r"\b(?:наушник\w*(?=.{0,30}\bкомплект\w*)|комплект\w*|комплектаци\w*)", re.I)),
    ("связь", re.compile(r"\b(?:связ\w*|wi-?fi|интернет\w*|сеть\w*)", re.I)),
    ("слот", re.compile(r"\b(?:слот\w*|разъ[её]м\w*|гнезд\w*|сим(?:-| )?карт\w*)", re.I)),
    ("размер_и_вес", re.compile(r"\b(?:размер\w*|габарит\w*|диагонал\w*|вес(?:а|у|ом|е)?|масс\w*)\b", re.I)),
    ("сканер отпечатка", re.compile(r"\b(?:сканер\w*|отпечат(?:ок|к\w*) пальц\w*)", re.I)),
    ("операционная система", re.compile(r"\b(?:андроид\w*|операционн\w* систем\w*|прошивк\w*)", re.I)),
)

SECTION_RE = re.compile(
    r"^Достоинства:\s*(?P<pros>.*?)\nНедостатки:\s*(?P<cons>.*?)(?=\n|$)",
    re.I | re.S,
)
PLACEHOLDER_RE = re.compile(
    r"^(?:в описании|нет|нету|не обнаружено|не замечено|отсутствуют|пока нет|не наш[её]л(?:а)?)\.?$",
    re.I,
)
COMPARATIVE_PRICE_RE = re.compile(r"\bза\s+такую\s+же\s+цен\w*\b", re.I)
DEPENDENT_QUALITY_RE = re.compile(
    r"\bкачеств(?:о|а|у|ом|е)?\s+(?:фото\w*|фотограф\w*|камер\w*|съ[её]мк\w*|видео\w*|сборк\w*)",
    re.I,
)
HEADPHONES_IN_BOX_RE = re.compile(r"\bнаушник\w*.{0,30}\bкомплект\w*", re.I)
NO_NEGATIVE_ASPECTS_RE = re.compile(
    r"(?:для\s+(?:этой|такой)\s+ценов\w+\s+категори\w+\s+их\s+нет|"
    r"(?:недостатк\w*|минус\w*)\s+(?:нет|нету|отсутств\w*|не\s+обнаруж\w*)|"
    r"не\s+считаю\s+недостатк\w*)\.?",
    re.I,
)


def extract_rows(review_id: int, text: str, url: str, rating: str) -> list[dict[str, str | int]]:
    """Return proposed explicit aspect mentions from an Otzovik review header."""
    match = SECTION_RE.search(text)
    if not match:
        return []

    rows: list[dict[str, str | int]] = []
    occupied_spans: list[tuple[int, int]] = []
    for group_name, sentiment in (("pros", "positive"), ("cons", "negative")):
        section = match.group(group_name).strip()
        if not section or PLACEHOLDER_RE.fullmatch(section):
            continue
        if sentiment == "negative" and NO_NEGATIVE_ASPECTS_RE.fullmatch(section):
            continue
        section_start = match.start(group_name)
        comparative_price_spans = [
            (section_start + item.start(), section_start + item.end())
            for item in COMPARATIVE_PRICE_RE.finditer(section)
        ]
        dependent_quality_spans = [
            (section_start + item.start(), section_start + item.end())
            for item in DEPENDENT_QUALITY_RE.finditer(section)
        ]
        has_headphones_in_box = bool(HEADPHONES_IN_BOX_RE.search(section))
        for normalized, pattern in ASPECT_PATTERNS:
            for aspect_match in pattern.finditer(section):
                start = section_start + aspect_match.start()
                end = section_start + aspect_match.end()
                if normalized == "цена" and any(
                    comparison_start <= start and end <= comparison_end
                    for comparison_start, comparison_end in comparative_price_spans
                ):
                    continue
                if normalized == "качество" and any(
                    quality_start <= start and end <= quality_end
                    for quality_start, quality_end in dependent_quality_spans
                ):
                    continue
                if (
                    normalized == "комплектация"
                    and aspect_match.group(0).lower().startswith("комплект")
                    and has_headphones_in_box
                ):
                    continue
                if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied_spans):
                    continue
                occupied_spans.append((start, end))
                rows.append(
                    {
                        "review_id": review_id,
                        "text": text,
                        "url": url,
                        "rating": rating,
                        "aspect": aspect_match.group(0),
                        "aspect_normalized": normalized,
                        "aspect_start": start,
                        "aspect_end": end,
                        "aspect_sentiment": sentiment,
                        "evidence_section": "Достоинства" if sentiment == "positive" else "Недостатки",
                        "annotation_status": "needs_review",
                        "annotation_source": "explicit_pros_cons_rule",
                        "annotator_note": "Проверьте аспект, границы и тональность перед использованием для обучения.",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    with args.input_csv.open(encoding="utf-8-sig", newline="") as input_file:
        source_rows = list(csv.DictReader(input_file))

    result_rows = [
        item
        for review_id, source_row in enumerate(source_rows, start=1)
        for item in extract_rows(
            review_id,
            source_row["text"],
            source_row["url"],
            source_row["rating"],
        )
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "review_id", "text", "url", "rating", "aspect", "aspect_normalized",
        "aspect_start", "aspect_end", "aspect_sentiment", "evidence_section",
        "annotation_status", "annotation_source", "annotator_note",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    covered_reviews = len({row["review_id"] for row in result_rows})
    print(f"Created {args.output_csv}: {len(result_rows)} candidate rows from {covered_reviews}/{len(source_rows)} reviews.")


if __name__ == "__main__":
    main()

"""Reproducible rule baseline for comparison with the ABSA classifier.

The original analysis assigned one review-level sentiment to every detected
aspect.  This module keeps that deliberately simple baseline while mapping its
terms to the same aspect taxonomy as the fine-tuned model.
"""

from __future__ import annotations

import re


# The vocabulary is frozen here so experimental results do not depend on a
# selected test file.  New labels must be added based on train/validation only.
LEGACY_TAXONOMY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
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
    ("комплектация", re.compile(r"\b(?:наушник\w*|комплект\w*|комплектаци\w*)", re.I)),
    ("связь", re.compile(r"\b(?:связ\w*|wi-?fi|вайфай|интернет\w*|сеть\w*)", re.I)),
    ("слот", re.compile(r"\b(?:слот\w*|разъ[её]м\w*|гнезд\w*|сим(?:-| )?карт\w*)", re.I)),
    ("размер_и_вес", re.compile(r"\b(?:размер\w*|габарит\w*|диагонал\w*|вес(?:а|у|ом|е)?|масс\w*)\b", re.I)),
    ("сканер отпечатка", re.compile(r"\b(?:сканер\w*|отпечат(?:ок|к\w*) пальц\w*)", re.I)),
    ("операционная система", re.compile(r"\b(?:андроид\w*|операционн\w* систем\w*|прошивк\w*)", re.I)),
)


def legacy_taxonomy(aspect: str) -> str | None:
    """Map an observed term to the legacy rule taxonomy, if it is covered."""
    return next((taxonomy for taxonomy, pattern in LEGACY_TAXONOMY_PATTERNS if pattern.search(aspect)), None)

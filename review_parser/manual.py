from __future__ import annotations

import re

import pandas as pd

from review_parser.extractors import looks_like_review, prepare_reviews


def build_reviews_from_text(raw_text: str, source: str = "manual-input") -> pd.DataFrame:
    chunks = re.split(r"\n\s*\n|(?:\r?\n){2,}|^\s*[-•]\s+", raw_text.strip(), flags=re.MULTILINE)
    reviews = [{"text": chunk.strip()} for chunk in chunks if looks_like_review(chunk.strip())]
    return prepare_reviews(reviews, source, limit=len(reviews))

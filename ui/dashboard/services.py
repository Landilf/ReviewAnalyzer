import io

import pandas as pd
import streamlit as st

from analysis_helpers.pipeline import analyze_reviews
from utils.data_loader import load_data


@st.cache_data(show_spinner="Выполняю анализ отзывов...")
def run_analysis(csv_bytes: bytes | None, filename: str | None, method: str, use_spacy_aspects: bool):
    data = _read_dataset(csv_bytes, filename)
    return analyze_reviews(data, method=method, use_spacy_aspects=use_spacy_aspects)


@st.cache_data(show_spinner="Выполняю анализ отзывов...")
def run_analysis_from_frame(data: pd.DataFrame, method: str, use_spacy_aspects: bool):
    return analyze_reviews(data, method=method, use_spacy_aspects=use_spacy_aspects)


def _read_dataset(csv_bytes: bytes | None, filename: str | None) -> pd.DataFrame:
    if csv_bytes is None:
        return load_data("reviews.csv")

    buffer = io.BytesIO(csv_bytes)
    if filename and filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

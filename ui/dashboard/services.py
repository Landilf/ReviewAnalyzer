import io

import pandas as pd

from analysis_helpers.pipeline import analyze_reviews, run_advanced_analysis


def run_analysis(
    csv_bytes: bytes | None,
    filename: str | None,
    method: str,
    cancel_check=None,
):
    data = _read_dataset(csv_bytes, filename)
    return analyze_reviews(data, method=method, cancel_check=cancel_check)


def run_analysis_from_frame(data: pd.DataFrame, method: str, cancel_check=None, progress_callback=None, include_advanced: bool = True):
    return analyze_reviews(
        data,
        method=method,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
        include_advanced=include_advanced,
    )


def run_advanced_analysis_for_result(result, cancel_check=None, progress_callback=None):
    return run_advanced_analysis(result, cancel_check=cancel_check, progress_callback=progress_callback)


def analyze_dataset(
    csv_bytes: bytes | None,
    filename: str | None,
    method: str,
    cancel_check=None,
):
    data = _read_dataset(csv_bytes, filename)
    return analyze_reviews(data, method=method, cancel_check=cancel_check)


def analyze_frame(
    data: pd.DataFrame,
    method: str,
    cancel_check=None,
):
    return analyze_reviews(data, method=method, cancel_check=cancel_check)


def _read_dataset(csv_bytes: bytes | None, filename: str | None) -> pd.DataFrame:
    if csv_bytes is None:
        raise ValueError("Загрузите CSV/XLSX-файл или импортируйте отзывы по ссылке.")

    buffer = io.BytesIO(csv_bytes)
    if filename and filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)

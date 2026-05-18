from dataclasses import dataclass

import pandas as pd


@dataclass
class ScrapeResult:
    reviews: pd.DataFrame
    source: str
    message: str = ""
    warning: str | None = None


import pandas as pd
from typing import Optional


def load_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

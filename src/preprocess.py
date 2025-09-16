from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns:
        date_parsed = pd.to_datetime(df['date'], errors='coerce', infer_datetime_format=True)
        df = df.copy()
        df['month'] = date_parsed.dt.month
        df['dayofweek'] = date_parsed.dt.dayofweek
        try:
            df['week'] = date_parsed.dt.isocalendar().week.astype('Int64')
        except Exception:
            df['week'] = date_parsed.dt.week
        df = df.drop(columns=['date'])
    return df


def encode_categoricals(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def preprocess(df: pd.DataFrame, target_col: str = 'actual_productivity') -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    df_proc = df.copy()

    if 'wip' in df_proc.columns:
        df_proc['wip'] = df_proc['wip'].fillna(df_proc['wip'].median())

    df_proc = engineer_time_features(df_proc)

    df_proc = encode_categoricals(df_proc, ['department', 'quarter', 'day'])

    y = df_proc[target_col] if target_col in df_proc.columns else None
    X = df_proc.drop(columns=[target_col], errors='ignore')

    X = X.select_dtypes(include=[np.number])
    skew = X.skew(numeric_only=True)
    log_cols = [c for c in X.columns if skew.get(c, 0) > 1.0 and (X[c] >= 0).all()]
    X_log = X.copy()
    for c in log_cols:
        X_log[c] = np.log1p(X_log[c])

    return X_log, y, df_proc, log_cols

# src/credit_risk/splits.py
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from pathlib import Path

def add_cv_folds(df: pd.DataFrame, y_col: str = "TARGET", n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    out = df.copy()
    out["fold"] = -1
    y = out[y_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(skf.split(out, y)):
        out.iloc[val_idx, out.columns.get_loc("fold")] = k
    return out

def split_holdout(df: pd.DataFrame, y_col: str = "TARGET", test_size: float = 0.2, seed: int = 42):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    y = df[y_col].values
    idx_tr, idx_va = next(sss.split(df, y))
    return df.iloc[idx_tr].copy(), df.iloc[idx_va].copy()

def save_folds_table(ids: pd.Series, fold: pd.Series, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": ids, "fold": fold}).to_parquet(path, index=False)

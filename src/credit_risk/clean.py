from __future__ import annotations
from typing import Sequence, Final
import numpy as np
import pandas as pd


# shared constants & utilities
SENTINEL_365243: Final[int] = 365243     # value found in days columns
_NUM_WINSOR_Q: Final[float] = 0.99        # cap numeric cols at 99th pct


# helpers 
def _replace_day_sentinels(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(SENTINEL_365243, np.nan)

def _fix_negative_money(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df.loc[df[c] <= 0, c] = np.nan

def _winsorise_numeric(df: pd.DataFrame, q: float = _NUM_WINSOR_Q) -> None:
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        return
    caps = df[num_cols].quantile(q)
    df[num_cols] = df[num_cols].clip(upper=caps, axis=1)

def _drop_quasi_constant(df: pd.DataFrame, tol: float = 0.99) -> None:
    if df.shape[1] == 0:
        return
    frac = df.apply(lambda s: s.value_counts(normalize=True, dropna=False).max() if len(s) else 0.0)
    to_drop = frac[frac >= tol].index.tolist()
    if to_drop:
        df.drop(columns=to_drop, inplace=True)

def _coerce_binary_flags(df: pd.DataFrame, prefix: str = "FLAG_") -> None:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return
    for c in cols:
        s = df[c]
        df[c] = np.where(s == 1, 1,
                 np.where(s == 0, 0,
                 np.where(s.astype(str).str.upper().isin({"Y","YES","TRUE"}), 1,
                 np.where(s.astype(str).str.upper().isin({"N","NO","FALSE"}), 0, np.nan))))

def _fix_nonnegative(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

def _enforce_nonpositive_days(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df.loc[df[c] > 0, c] = np.nan

# Cleaning Tables
def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "SK_ID_CURR" in out.columns:
        out.drop_duplicates(subset=["SK_ID_CURR"], inplace=True)

    _replace_day_sentinels(out, ["DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"])
    _enforce_nonpositive_days(out, ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION"])
    if "DAYS_BIRTH" in out.columns:
        out.loc[(out["DAYS_BIRTH"] > -18 * 365) | (out["DAYS_BIRTH"] < -100 * 365), "DAYS_BIRTH"] = np.nan

    _fix_negative_money(out, ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY"])
    _fix_nonnegative(out, ["CNT_CHILDREN", "CNT_FAM_MEMBERS"])
    _coerce_binary_flags(out, prefix="FLAG_")
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

def clean_bureau(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    _replace_day_sentinels(out, ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"])
    _enforce_nonpositive_days(out, ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT"])
    _fix_nonnegative(out, ["CNT_CREDIT_PROLONG", "CREDIT_DAY_OVERDUE"])
    _fix_negative_money(out, ["AMT_CREDIT_SUM_OVERDUE"])
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

def clean_prev(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    _enforce_nonpositive_days(out, [
        "DAYS_DECISION", "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION"
    ])
    _coerce_binary_flags(out, prefix="FLAG_")
    _coerce_binary_flags(out, prefix="NFLAG")
    _fix_negative_money(out, ["AMT_APPLICATION", "AMT_CREDIT", "AMT_ANNUITY"])
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

def clean_pos(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    _replace_day_sentinels(out, ["MONTHS_BALANCE"])
    _fix_nonnegative(out, ["CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"])
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

def clean_installments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    _replace_day_sentinels(out, ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
    _enforce_nonpositive_days(out, ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
    _fix_nonnegative(out, ["NUM_INSTALMENT_NUMBER", "NUM_INSTALMENT_VERSION"])
    _fix_negative_money(out, ["AMT_PAYMENT", "AMT_INSTALMENT"])
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

def clean_cc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    _replace_day_sentinels(out, ["MONTHS_BALANCE"])
    _fix_nonnegative(out, ["SK_DPD", "SK_DPD_DEF"])
    _fix_negative_money(out, ["AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_CURRENT"])
    _winsorise_numeric(out)
    _drop_quasi_constant(out)
    return out

__all__ = [
    "SENTINEL_365243",
    "clean_application",
    "clean_bureau",
    "clean_prev",
    "clean_pos",
    "clean_installments",
    "clean_cc",
]
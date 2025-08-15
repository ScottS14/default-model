from __future__ import annotations
import numpy as np
import pandas as pd

def compute_app_ratios(app: pd.DataFrame) -> pd.DataFrame:
    out = app.copy()

    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        b = b.replace(0, np.nan)
        s = a / b
        return s.replace([np.inf, -np.inf], np.nan)

    # Years from day counters
    if {"DAYS_BIRTH"}.issubset(out.columns):
        out["DAYS_BIRTH_YEARS"] = (-out["DAYS_BIRTH"] / 365.25)
    if {"DAYS_EMPLOYED"}.issubset(out.columns):
        out["DAYS_EMPLOYED_YEARS"] = (out["DAYS_EMPLOYED"] / 365.25)

    # Core credit ratios (only if inputs exist)
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(out.columns):
        out["INCOME_CREDIT_RATIO"] = _safe_div(out["AMT_INCOME_TOTAL"],
                                                out["AMT_CREDIT"])
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANNUITY_INCOME_RATIO"] = _safe_div(out["AMT_ANNUITY"],
                                                 out["AMT_INCOME_TOTAL"])
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(out.columns):
        out["ANNUITY_CREDIT_RATIO"] = _safe_div(out["AMT_ANNUITY"],
                                                 out["AMT_CREDIT"])
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(out.columns):
        out["CREDIT_GOODS_RATIO"] = _safe_div(out["AMT_CREDIT"],
                                               out["AMT_GOODS_PRICE"])
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(out.columns):
        out["EMPLOY_TO_AGE_RATIO"] = _safe_div(out["DAYS_EMPLOYED"],
                                                out["DAYS_BIRTH"])
    if {"AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"}.issubset(out.columns):
        out["INCOME_PER_PERSON"] = _safe_div(out["AMT_INCOME_TOTAL"],
                                             out["CNT_FAM_MEMBERS"]
                                             .replace(0, np.nan))

    return out
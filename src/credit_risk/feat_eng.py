from __future__ import annotations
import numpy as np
import pandas as pd

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        b = b.replace(0, np.nan)
        s = a / b
        return s.replace([np.inf, -np.inf], np.nan)

def compute_app_ratios(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"DAYS_BIRTH"}.issubset(out.columns):
        out["DAYS_BIRTH_YEARS"] = (-out["DAYS_BIRTH"] / 365.25).astype("float32")

    if {"DAYS_EMPLOYED"}.issubset(out.columns):
        out["DAYS_EMPLOYED_YEARS"] = (out["DAYS_EMPLOYED"] / 365.25).astype("float32")

    if {"AMT_INCOME_TOTAL","AMT_CREDIT"}.issubset(out.columns):
        out["DEBT_TO_INCOME"] = _safe_div(out["AMT_CREDIT"],
                                           out["AMT_INCOME_TOTAL"])
        
    if {"AMT_GOODS_PRICE","AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["GOODS_TO_INCOME"] = _safe_div(out["AMT_GOODS_PRICE"],
                                            out["AMT_INCOME_TOTAL"])
        
    if {"AMT_ANNUITY","AMT_INCOME_TOTAL"}.issubset(out.columns):
        out["ANN_TO_INC_PC"] = _safe_div(out["AMT_ANNUITY"], 
                                         out["AMT_INCOME_TOTAL"]) * out.get("CNT_FAM_MEMBERS", 1)
        
    if {"AMT_GOODS_PRICE","AMT_CREDIT"}.issubset(out.columns):
        out["DOWNPAY_SHARE"] = _safe_div((out["AMT_GOODS_PRICE"] - out["AMT_CREDIT"]),
                                          out["AMT_GOODS_PRICE"]).clip(0,1)
        
    if {"AMT_CREDIT","AMT_ANNUITY"}.issubset(out.columns):
        out["IMPLIED_TERM"] = _safe_div(out["AMT_CREDIT"], out["AMT_ANNUITY"])

    if {"AMT_INCOME_TOTAL","CNT_CHILDREN"}.issubset(out.columns):
        out["INC_PER_DEP"] = _safe_div(out["AMT_INCOME_TOTAL"], 
                                       (out["CNT_CHILDREN"] + 1))

    if {"DAYS_ID_PUBLISH","DAYS_REGISTRATION"}.issubset(out.columns):
        out["DOC_RECENCY_RATIO"] = _safe_div(out["DAYS_ID_PUBLISH"].abs(),
                                              out["DAYS_REGISTRATION"].abs())
        
    if {"DAYS_LAST_PHONE_CHANGE","DAYS_REGISTRATION"}.issubset(out.columns):
        out["PHONE_RECENCY_RATIO"] = _safe_div(out["DAYS_LAST_PHONE_CHANGE"].abs(),
                                                out["DAYS_REGISTRATION"].abs())

    return out
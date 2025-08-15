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

def aggregate_bureau(bureau: pd.DataFrame, bb: pd.DataFrame) -> pd.DataFrame:
    bb_counts = (
        bb.groupby("SK_ID_BUREAU")["STATUS"]
          .value_counts()
          .unstack(fill_value=0)
          .add_prefix("BB_STATUS_")
          .reset_index()
    )
    b = bureau.merge(bb_counts, on="SK_ID_BUREAU", how="left")

    bb_cols = [c for c in b.columns if c.startswith("BB_STATUS_")]

    agg_dict = {
        "AMT_CREDIT_SUM": ["mean", "sum", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "sum"],
        "DAYS_CREDIT": ["min", "max", "mean"],
    }
    # add SK_DPD if present
    for c in ["SK_DPD", "SK_DPD_DEF"]:
        if c in b.columns:
            agg_dict[c] = ["mean", "max"]

    # build final dict including bb status sums
    final_agg = agg_dict | {c: "sum" for c in bb_cols}

    agg = b.groupby("SK_ID_CURR", observed=True).agg(final_agg)
    # flatten multiindex columns
    agg.columns = ["BUREAU_" + "_".join(map(str, col)).upper() for col in agg.columns.to_flat_index()]
    agg = agg.reset_index()

    # cast numeric to dtype
    num_cols = agg.select_dtypes(include="number").columns
    agg[num_cols] = agg[num_cols]
    return agg

def aggregate_prev(prev: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    p = prev.copy()
    if {"AMT_APPLICATION", "AMT_CREDIT"}.issubset(p.columns):
        p["APP_CREDIT_PERC"] = _safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(p.columns):
        p["GOODS_CREDIT_PERC"] = _safe_div(p["AMT_GOODS_PRICE"], p["AMT_CREDIT"])

    agg_dict = {
        "AMT_CREDIT": ["mean", "max"],
        "AMT_ANNUITY": ["mean"],
        "APP_CREDIT_PERC": ["mean"],
        "DAYS_DECISION": ["min", "max"],
    }
    if "NAME_CONTRACT_STATUS" in p.columns:
        agg_dict["NAME_CONTRACT_STATUS"] = ["nunique"]

    agg = p.groupby("SK_ID_CURR", observed=True).agg(agg_dict)
    agg.columns = ["PREV_" + "_".join(map(str, c)).upper() for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg[agg.select_dtypes("number").columns] = agg.select_dtypes("number").astype(dtype)
    return agg


def aggregate_installments(ins: pd.DataFrame, *, dtype: str = "float32") -> pd.DataFrame:
    i = ins.copy()
    if {"AMT_PAYMENT", "AMT_INSTALMENT"}.issubset(i.columns):
        i["PAYMENT_DIFF"] = i["AMT_PAYMENT"] - i["AMT_INSTALMENT"]
        i["PAYMENT_PERC"] = _safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    if {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(i.columns):
        i["LATE_DAYS"] = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]

    agg = i.groupby("SK_ID_CURR", observed=True).agg({
        "PAYMENT_DIFF": ["mean", "sum", "max"],
        "PAYMENT_PERC": ["mean", "max"],
        "LATE_DAYS": ["mean", "max"],
    })
    agg.columns = ["INS_" + "_".join(map(str, c)).upper() for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg[agg.select_dtypes("number").columns] = agg.select_dtypes("number")
    return agg


def aggregate_pos(pos: pd.DataFrame) -> pd.DataFrame:
    g = pos.groupby("SK_ID_CURR", observed=True).agg({
        "SK_DPD": ["mean", "max"] if "SK_DPD" in pos.columns else [],
        "SK_DPD_DEF": ["mean", "max"] if "SK_DPD_DEF" in pos.columns else [],
        "CNT_INSTALMENT_FUTURE": ["mean", "min"] if "CNT_INSTALMENT_FUTURE" in pos.columns else [],
    })
    # drop any empty aggs
    g = g.loc[:, g.columns.get_level_values(0) != ""]
    g.columns = ["POS_" + "_".join(map(str, c)).upper() for c in g.columns.to_flat_index()]
    g = g.reset_index()
    g[g.select_dtypes("number").columns] = g.select_dtypes("number")
    return g


def aggregate_cc(cc: pd.DataFrame) -> pd.DataFrame:
    c = cc.copy()
    if {"AMT_DRAWINGS_CURRENT", "AMT_CREDIT_LIMIT_ACTUAL"}.issubset(c.columns):
        c["DRAW_RATIO"] = _safe_div(c["AMT_DRAWINGS_CURRENT"], c["AMT_CREDIT_LIMIT_ACTUAL"])

    agg = c.groupby("SK_ID_CURR", observed=True).agg({
        "AMT_BALANCE": ["mean", "max"] if "AMT_BALANCE" in c.columns else [],
        "AMT_PAYMENT_TOTAL_CURRENT": ["mean", "sum"] if "AMT_PAYMENT_TOTAL_CURRENT" in c.columns else [],
        "DRAW_RATIO": ["mean", "max"] if "DRAW_RATIO" in c.columns else [],
        "SK_DPD": ["mean", "max"] if "SK_DPD" in c.columns else [],
    })
    agg = agg.loc[:, agg.columns.get_level_values(0) != ""]
    agg.columns = ["CC_" + "_".join(map(str, c)).upper() for c in agg.columns.to_flat_index()]
    agg = agg.reset_index()
    agg[agg.select_dtypes("number").columns] = agg.select_dtypes("number")
    return agg
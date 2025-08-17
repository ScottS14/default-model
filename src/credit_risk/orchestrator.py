from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from credit_risk.clean import (
    clean_application, clean_bureau, clean_prev, clean_pos, clean_installments, clean_cc
)
from credit_risk.feat_eng import (
    compute_app_ratios, aggregate_bureau, aggregate_prev,
    aggregate_pos, aggregate_installments, aggregate_cc
)

DF = pd.DataFrame

def build_features(app_clean: DF, *, frames: dict[str, DF], include: tuple[str, ...]) -> DF:
    X = app_clean.copy()

    if "app_ratios" in include:
        X = compute_app_ratios(X)

    if "bureau" in include and {"bureau","bb"} <= frames.keys():
        X = X.merge(aggregate_bureau(frames["bureau"], frames["bb"]), on="SK_ID_CURR", how="left")
    if "previous" in include and "prev" in frames:
        X = X.merge(aggregate_prev(frames["prev"]), on="SK_ID_CURR", how="left")
    if "install" in include and "ins" in frames:
        X = X.merge(aggregate_installments(frames["ins"]), on="SK_ID_CURR", how="left")
    if "pos" in include and "pos" in frames:
        X = X.merge(aggregate_pos(frames["pos"]), on="SK_ID_CURR", how="left")
    if "cc" in include and "cc" in frames:
        X = X.merge(aggregate_cc(frames["cc"]), on="SK_ID_CURR", how="left")

    return X

def ohe_for_xgb(df: DF) -> DF:
    cat_cols = df.select_dtypes(include=["object","category","bool"]).columns
    return pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=False)

def main(raw_dir: str, out_dir: str, include: list[str]) -> None:
    raw = Path(raw_dir)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # load raw 
    app_train = pd.read_csv(raw / "application_train.csv")
    app_test  = pd.read_csv(raw / "application_test.csv")
    bureau    = pd.read_csv(raw / "bureau.csv")
    bb        = pd.read_csv(raw / "bureau_balance.csv")
    prev      = pd.read_csv(raw / "previous_application.csv")
    pos       = pd.read_csv(raw / "POS_CASH_balance.csv")
    ins       = pd.read_csv(raw / "installments_payments.csv")
    cc        = pd.read_csv(raw / "credit_card_balance.csv")

    # clean each table 
    app_tr_c = clean_application(app_train)
    app_te_c = clean_application(app_test)

    bureau_c = clean_bureau(bureau)
    prev_c   = clean_prev(prev)
    pos_c    = clean_pos(pos)
    ins_c    = clean_installments(ins)
    cc_c     = clean_cc(cc)

    frames = {"bureau": bureau_c, "bb": bb, "prev": prev_c, "pos": pos_c, "ins": ins_c, "cc": cc_c}

    # build features (toggle tables via include)
    include_tup = tuple(include)  
    X_train = build_features(app_tr_c, frames=frames, include=include_tup)
    X_test  = build_features(app_te_c, frames=frames, include=include_tup)

    #  split out target if present 
    y = X_train["TARGET"] if "TARGET" in X_train.columns else None
    if y is not None:
        X_train = X_train.drop(columns=["TARGET"])

    #  LGBM matrices: keep categoricals as is 
    lgb_train = X_train.copy()
    lgb_test  = X_test.copy()
    if y is not None:
        lgb_train["TARGET"] = y.values

    #  XGB matrices: OHE and align 
    xgb_train = ohe_for_xgb(X_train)
    xgb_test  = ohe_for_xgb(X_test)
    xgb_train, xgb_test = xgb_train.align(xgb_test, join="outer", axis=1, fill_value=0)
    # ensure TARGET only on train
    if y is not None:
        xgb_train["TARGET"] = y.values
        xgb_test = xgb_test.drop(columns=["TARGET"], errors="ignore")

    #  save parquet files 
    (outp / "train_features_lgbm.parquet").write_bytes(lgb_train.to_parquet(index=False))
    (outp / "test_features_lgbm.parquet").write_bytes(lgb_test.to_parquet(index=False))
    (outp / "train_features_xgb.parquet").write_bytes(xgb_train.to_parquet(index=False))
    (outp / "test_features_xgb.parquet").write_bytes(xgb_test.to_parquet(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/home_credit")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument(
        "--include", nargs="*", default=["app_ratios","bureau","previous","install","pos","cc"],
        help="Feature blocks to include (e.g. app_ratios bureau previous)"
    )
    args = p.parse_args()
    main(args.raw_dir, args.out_dir, args.include)

from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from clean import (
    clean_application, clean_bureau, clean_prev, clean_pos, clean_installments, 
    clean_cc
)
from feat_eng import (
    compute_app_ratios, aggregate_bureau, aggregate_prev,
    aggregate_pos, aggregate_installments, aggregate_cc
)
from splits import add_cv_folds, save_folds_table

DF = pd.DataFrame

# Build features
def build_features(app_clean: DF, *, frames: dict[str, DF], include: tuple[str, ...]) -> DF:
    X = app_clean.copy()

    if "app_ratios" in include:
        X = compute_app_ratios(X)

    if "bureau" in include and {"bureau", "bb"} <= frames.keys():
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

# Model-specific views
def cast_lgbm_categoricals(df: DF) -> DF:
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "bool"]).columns
    for c in cat_cols:
        out[c] = out[c].astype("category")
    return out

def ohe_for_xgb(df: DF) -> DF:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    # keep all levels; model will handle regularization
    return pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=False)

# CLI main
def main(raw_dir: str, out_dir: str, include: list[str]) -> None:
    raw = Path(raw_dir)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Load raw
    app_train = pd.read_csv(raw / "application_train.csv")
    app_test  = pd.read_csv(raw / "application_test.csv")
    bureau    = pd.read_csv(raw / "bureau.csv")
    bb        = pd.read_csv(raw / "bureau_balance.csv")
    prev      = pd.read_csv(raw / "previous_application.csv")
    pos       = pd.read_csv(raw / "POS_CASH_balance.csv")
    ins       = pd.read_csv(raw / "installments_payments.csv")
    cc        = pd.read_csv(raw / "credit_card_balance.csv")

    # Clean each table
    app_tr_c = clean_application(app_train)
    app_te_c = clean_application(app_test)
    frames = {
        "bureau": clean_bureau(bureau),
        "bb":     bb,
        "prev":   clean_prev(prev),
        "pos":    clean_pos(pos),
        "ins":    clean_installments(ins),
        "cc":     clean_cc(cc),
    }

    # Build features (toggle via --include)
    include_tup = tuple(include)
    X_train = build_features(app_tr_c, frames=frames, include=include_tup)
    X_test  = build_features(app_te_c, frames=frames, include=include_tup)

    # Split out TARGET
    y = X_train["TARGET"] if "TARGET" in X_train.columns else None
    if y is not None:
        X_train = X_train.drop(columns=["TARGET"])

    # LGBM view
    lgb_train = cast_lgbm_categoricals(X_train)
    lgb_test  = cast_lgbm_categoricals(X_test)
    if y is not None:
        lgb_train["TARGET"] = y.values

    # persist
    lgb_train.to_parquet(outp / "train_features_lgbm.parquet", index=False)
    lgb_test.to_parquet(outp / "test_features_lgbm.parquet", index=False)

    # add reusable folds (only on train)
    if y is not None:
        lgb_with_folds = add_cv_folds(lgb_train, y_col="TARGET", n_splits=5, seed=42)
        lgb_with_folds.to_parquet(outp / "train_with_folds_lgbm.parquet", index=False)
        # save a small fold map keyed by SK_ID_CURR
        if "SK_ID_CURR" in lgb_with_folds.columns:
            save_folds_table(
                ids=lgb_with_folds["SK_ID_CURR"], 
                fold=lgb_with_folds["fold"], 
                path=outp / "folds_by_sk_id.parquet"
            )

    # XGB view 
    xgb_train = ohe_for_xgb(X_train)
    xgb_test  = ohe_for_xgb(X_test)
    xgb_train, xgb_test = xgb_train.align(xgb_test, join="outer", axis=1, fill_value=0)

    # ensure TARGET only on train
    if y is not None:
        xgb_train["TARGET"] = y.values
        xgb_test.drop(columns=["TARGET"], errors="ignore", inplace=True)

    # persist + save OHE columns for reproducibility
    xgb_train.to_parquet(outp / "train_features_xgb.parquet", index=False)
    xgb_test.to_parquet(outp / "test_features_xgb.parquet", index=False)
    (outp / "xgb_columns.json").write_text(
        json.dumps({"columns": list(xgb_train.drop(columns=["TARGET"], 
                                                   errors="ignore").columns)}, 
                                                   indent=2)
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/raw/home_credit")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument(
        "--include", nargs="*", default=["app_ratios", "bureau", "previous", 
                                         "install", "pos", "cc"],
        help="Feature blocks to include (e.g. app_ratios bureau previous)"
    )
    args = p.parse_args()
    main(args.raw_dir, args.out_dir, args.include)

from __future__ import annotations
import argparse, json, os, pathlib
from typing import Tuple
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None


def _unwrap_best_params(p: dict) -> dict:
    return p.get("best_params", p)


def _sanitize_lgbm_params(p: dict) -> Tuple[dict, int]:
    p = _unwrap_best_params(p).copy()

    num_boost_round = int(p.pop("n_estimators", 1000))
    if "n_jobs" in p:
        p["num_threads"] = int(p.pop("n_jobs"))
    # remove stuff lgb.train() doesn't understand
    for k in ("eval_metric", "best_score", "best_value", "best_trial_number"):
        p.pop(k, None)
    # xgb-only keys if they leaked
    for k in ("tree_method", "gamma"):
        p.pop(k, None)
    p.setdefault("objective", "binary")
    return p, num_boost_round


def _sanitize_xgb_params(p: dict) -> Tuple[dict, int]:
    p = _unwrap_best_params(p).copy()
    num_boost_round = int(p.pop("n_estimators", p.pop("num_boost_round", 1000)))
    return p, num_boost_round


def _ensure_parent(path: str | os.PathLike) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def train_lgbm(X: pd.DataFrame, y: np.ndarray, params_json: str, out_model: str) -> None:
    assert lgb is not None, "lightgbm is not installed."
    with open(params_json, "r") as f:
        raw = json.load(f)
    params, num_boost_round = _sanitize_lgbm_params(raw)

    # cast categoricals for lgbm
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "bool") or isinstance(X[c].dtype, pd.CategoricalDtype)]
    Xc = X.copy()
    for c in cat_cols:
        if not isinstance(Xc[c].dtype, pd.CategoricalDtype):
            Xc[c] = Xc[c].astype("category")

    dtrain = lgb.Dataset(Xc, label=y, categorical_feature=cat_cols, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)

    _ensure_parent(out_model)
    booster.save_model(out_model)

    # persist feature order for later scoring/SHAP
    with open(os.path.splitext(out_model)[0] + "_feature_list.json", "w") as f:
        json.dump(X.columns.tolist(), f)
    print(f"Saved LightGBM model → {out_model}")


def train_xgb(X: pd.DataFrame, y: np.ndarray, params_json: str, out_model: str) -> None:
    assert xgb is not None, "xgboost is not installed."
    with open(params_json, "r") as f:
        raw = json.load(f)
    params, num_boost_round = _sanitize_xgb_params(raw)

    dtrain = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)

    _ensure_parent(out_model)
    booster.save_model(out_model)

    with open(os.path.splitext(out_model)[0] + "_feature_list.json", "w") as f:
        json.dump(X.columns.tolist(), f)
    print(f"Saved XGBoost model → {out_model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", required=True, help="train features parquet (with TARGET)")
    ap.add_argument("--target-col", default="TARGET")
    ap.add_argument("--model-kind", choices=["lgbm", "xgb"], required=True)
    ap.add_argument("--best-params-json", required=True)
    ap.add_argument("--out-model", help="where to save the trained model")
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    assert args.target_col in df.columns, f"{args.target_col} missing in features"
    y = df[args.target_col].astype(int).values
    X = df.drop(columns=[args.target_col])

    if args.model_kind == "lgbm":
        out = args.out_model or "src/credit_risk/models/lgbm/best_lgbm.txt"
        train_lgbm(X, y, args.best_params_json, out)
    else:
        out = args.out_model or "src/credit_risk/models/xgb/best_xgb.json"
        train_xgb(X, y, args.best_params_json, out)


if __name__ == "__main__":
    main()

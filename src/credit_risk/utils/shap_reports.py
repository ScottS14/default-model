from __future__ import annotations
import argparse, json, os, random
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

def _load_best_params(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def _fit_lgbm(X: pd.DataFrame, y: np.ndarray, params: dict):
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("category", "object", "bool")]
    dtr = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)
    booster = lgb.train(params, dtr, num_boost_round=int(params.get("n_estimators", 1000)))
    return booster

def _fit_xgb(X: pd.DataFrame, y: np.ndarray, params: dict):
    dtr = xgb.DMatrix(X.values, label=y, feature_names=X.columns.tolist())
    booster = xgb.train(params, dtr, num_boost_round=int(params.get("n_estimators", 1000)))
    return booster

def _predict(model, X: pd.DataFrame) -> np.ndarray:
    # LightGBM Booster
    if lgb is not None and isinstance(model, lgb.Booster):
        return model.predict(X)
    # XGBoost Booster
    if xgb is not None and isinstance(model, xgb.Booster):
        d = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
        return model.predict(d)
    # sklearn-wrapped LGBM
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)

def _tree_explainer(model):
    return shap.TreeExplainer(model)

def _as_positive_class_shap(explainer, X_sample):
    sv = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(sv, list) and len(sv) == 2:
        return sv[1]
    return sv

def shap_beeswarm_and_force(model, X: pd.DataFrame, out_dir: str, k_force: int = 5, sample_beeswarm: int = 3000, seed: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # sample for beeswarm (speed & readability)
    if len(X) > sample_beeswarm:
        idx = rng.choice(len(X), size=sample_beeswarm, replace=False)
        X_swarm = X.iloc[idx].copy()
    else:
        X_swarm = X

    explainer = _tree_explainer(model)
    sv = _as_positive_class_shap(explainer, X_swarm)

    # Beeswarm
    shap.summary_plot(sv, X_swarm, show=False)
    plt.title("SHAP summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=150)
    plt.close()

    # pick 5 customers: highest predicted risk
    probs = _predict(model, X)
    top_idx = np.argsort(-probs)[:k_force]

    for rank, i in enumerate(top_idx, start=1):
        x_row = X.iloc[i:i+1]
        sv_row = _as_positive_class_shap(explainer, x_row)
        shap.force_plot(
            explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[1],
            sv_row[0] if isinstance(sv_row, np.ndarray) else sv_row,
            x_row,
            matplotlib=True,
            show=False,
        )
        plt.title(f"SHAP force plot — customer {i} (rank {rank})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_force_customer_{i}.png"), dpi=150)
        plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-parquet", required=True, help="Parquet with features (and optionally TARGET)")
    p.add_argument("--target-col", default="TARGET")
    p.add_argument("--model-kind", choices=["lgbm","xgb"], required=True)
    p.add_argument("--model-file", help="Optional: pre-trained model file (lgbm.txt / xgb.json)")
    p.add_argument("--best-params-json", help="If no model file, train with these params")
    p.add_argument("--out-dir", default="reports/figures/shap")
    p.add_argument("--k-force", type=int, default=5)
    p.add_argument("--sample-beeswarm", type=int, default=3000)
    args = p.parse_args()

    Xy = pd.read_parquet(args.features_parquet)
    y = None
    if args.target_col in Xy.columns:
        y = Xy[args.target_col].astype(int).values
        X = Xy.drop(columns=[args.target_col])
    else:
        X = Xy

    # Load or train model
    if args.model_file:
        if args.model-kind == "lgbm":
            booster = lgb.Booster(model_file=args.model_file)
        else:
            booster = xgb.Booster()
            booster.load_model(args.model_file)
        model = booster
    else:
        assert y is not None, "Need TARGET to train with best params."
        params = _load_best_params(args.best_params_json)
        if args.model_kind == "lgbm":
            model = _fit_lgbm(X, y, params)
        else:
            model = _fit_xgb(X, y, params)

    shap_beeswarm_and_force(model, X, out_dir=args.out_dir, k_force=args.k_force, sample_beeswarm=args.sample_beeswarm)

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, json, os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import xgboost as xgb
except Exception:
    xgb = None


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    if lgb is not None and isinstance(model, lgb.Booster):
        Xc = X.copy()
        for c in Xc.columns:
            if str(Xc[c].dtype) in ("object", "bool"):
                Xc[c] = Xc[c].astype("category")
        return model.predict(Xc)
    if xgb is not None and isinstance(model, xgb.Booster):
        d = xgb.DMatrix(X.values, feature_names=X.columns.tolist())
        return model.predict(d)
    raise TypeError("Unsupported model type for prediction.")


def _tree_explainer(model):
    return shap.TreeExplainer(model)


def _pos_class_sv(explainer, X):
    sv = explainer.shap_values(X, check_additivity=False)
    return sv[1] if isinstance(sv, list) and len(sv) == 2 else sv


def _load_feature_list_if_any(model_file: str) -> list[str] | None:
    base, _ = os.path.splitext(model_file)
    meta = base + "_feature_list.json"
    if os.path.exists(meta):
        with open(meta, "r") as f:
            return json.load(f)
    return None


def shap_beeswarm_and_individual(
    model, X: pd.DataFrame, out_dir: str,
    k: int = 5, beeswarm_sample: int = 3000, style: str = "waterfall"
):
    os.makedirs(out_dir, exist_ok=True)

    # Beeswarm on a sample for readability
    if len(X) > beeswarm_sample:
        idx = np.random.default_rng(42).choice(len(X), size=beeswarm_sample, replace=False)
        X_swarm = X.iloc[idx].copy()
    else:
        X_swarm = X

    explainer = _tree_explainer(model)
    sv_swarm = _pos_class_sv(explainer, X_swarm)

    shap.summary_plot(sv_swarm, X_swarm, show=False)
    plt.title("SHAP summary (beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=160)
    plt.close()

    # Top-k highest risk cases
    probs = _predict(model, X)
    top_idx = np.argsort(-probs)[:k]

    # Individual plots: prefer waterfall for legibility
    for rank, i in enumerate(top_idx, 1):
        xrow = X.iloc[i:i+1]
        sv_row = _pos_class_sv(explainer, xrow)

        plt.figure(figsize=(8, 6))
        if style == "force":
            shap.force_plot(
                explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[1],
                sv_row[0] if isinstance(sv_row, np.ndarray) else sv_row,
                xrow,
                matplotlib=True, show=False
            )
            plt.title(f"Force plot — row {i} (rank {rank})")
        else:
            # waterfall (clear static viz)
            shap.plots._waterfall.waterfall_legacy(
                explainer.expected_value if not isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value[1],
                sv_row[0] if isinstance(sv_row, np.ndarray) else sv_row,
                feature_names=xrow.columns.tolist(),
                max_display=20,
                show=False
            )
            plt.title(f"Waterfall — row {i} (rank {rank})")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"shap_{style}_row_{i}.png"), dpi=160)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-parquet", required=True)
    p.add_argument("--model-kind", choices=["lgbm","xgb"], required=True)
    p.add_argument("--model-file", required=True, help="best_lgbm.txt or best_xgb.json")
    p.add_argument("--out-dir", default="reports/figures/shap")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--beeswarm-sample", type=int, default=3000)
    p.add_argument("--style", choices=["waterfall","force"], default="waterfall")
    args = p.parse_args()

    X = pd.read_parquet(args.features_parquet)
    # remove TARGET if present
    if "TARGET" in X.columns:
        X = X.drop(columns=["TARGET"])

    # align columns to the exact training order if metadata file exists
    feat_list = _load_feature_list_if_any(args.model_file)
    if feat_list is not None:
        # some columns might be missing; use intersection in correct order
        keep = [c for c in feat_list if c in X.columns]
        X = X.loc[:, keep]

    # load model
    if args.model_kind == "lgbm":
        assert lgb is not None, "lightgbm not installed"
        model = lgb.Booster(model_file=args.model_file)
    else:
        assert xgb is not None, "xgboost not installed"
        model = xgb.Booster()
        model.load_model(args.model_file)

    shap_beeswarm_and_individual(
        model, X, out_dir=args.out_dir, k=args.k,
        beeswarm_sample=args.beeswarm_sample, style=args.style
    )


if __name__ == "__main__":
    main()

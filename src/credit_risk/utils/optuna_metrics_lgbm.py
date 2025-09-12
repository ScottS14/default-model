from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import mlflow
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve, f1_score, 
    average_precision_score,
)

from credit_risk.utils.optuna_common import log_study_figures, export_trials_csv

# Reporting path
report_base = os.path.join("reports", "lgbm")

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def _report_path(*parts: str) -> str:
    path = os.path.join(report_base, *parts)
    _ensure_dir(os.path.dirname(path))
    return path

def _find_cv_metric_keys(cv_res: dict):
    ap_mean = "valid average_precision-mean"
    ap_std = "valid average_precision-stdv"
    if ap_mean in cv_res:
        return ap_mean, (ap_std if ap_std in cv_res else None), "AUC-PR (Average Precision)"
    raise KeyError(f"AUC-PR metric not found; keys: {list(cv_res.keys())}")

def lgbm_average_precision(preds, train_data):
    y_true = train_data.get_label()
    ap = average_precision_score(y_true, preds)
    return "average_precision", ap, True

def log_cv_curve(cv_res, name_prefix="cv"):
    mean_key, std_key, pretty = _find_cv_metric_keys(cv_res)
    vals = np.asarray(cv_res[mean_key], dtype=float)
    stds = np.asarray(cv_res.get(std_key, np.zeros_like(vals)), dtype=float)
    xs = np.arange(1, len(vals) + 1, dtype=int)

    plt.figure()
    plt.plot(xs, vals, label=f"{pretty} (mean)")
    if stds.shape == vals.shape and np.all(np.isfinite(stds)):
        plt.fill_between(xs, vals - stds, vals + stds, alpha=0.2, label="±1 std")
    plt.xlabel("Boosting round"); plt.ylabel(pretty)
    plt.title(f"{name_prefix} {pretty} over iterations"); plt.legend()

    # Save locally under reports/lgbm and also log to MLflow
    local_path = _report_path("figures", f"{name_prefix}_cv_metric_curve.png")
    plt.savefig(local_path, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), f"figures/{name_prefix}_cv_metric_curve.png")
    plt.close()

def train_oof_and_log(best_params, X, y, folds, cat_cols):
    oof = np.zeros(len(y), dtype=float)
    with mlflow.start_run(run_name="oof_evaluation", nested=True):
        for fi, (trn_idx, val_idx) in enumerate(folds):
            dtr = lgb.Dataset(X.iloc[trn_idx], label=y[trn_idx], categorical_feature=cat_cols, free_raw_data=False)
            dvl = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], categorical_feature=cat_cols, free_raw_data=False)

            booster = lgb.train(
                best_params, dtr, num_boost_round=5000,
                valid_sets=[dvl], valid_names=["valid"],
                feval=lgbm_average_precision,
                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
            )
            preds = booster.predict(X.iloc[val_idx], num_iteration=booster.best_iteration)
            oof[val_idx] = preds

            mlflow.log_metric(f"fold{fi}_auc", float(roc_auc_score(y[val_idx], preds)))
            mlflow.log_metric(f"fold{fi}_aucpr", float(average_precision_score(y[val_idx], preds)))

        # OOF metrics/plots
        oof_auc = roc_auc_score(y, oof)
        oof_ap = average_precision_score(y, oof)
        precision, recall, thr_pr = precision_recall_curve(y, oof)
        fprs, tprs, _ = roc_curve(y, oof)
        f1s = [f1_score(y, (oof >= t).astype(int)) for t in thr_pr[:-1]] if len(thr_pr) > 1 else [0.0]
        best_thr = float(thr_pr[int(np.argmax(f1s))]) if len(thr_pr) > 1 else 0.5

        mlflow.log_metric("oof_auc", float(oof_auc))
        mlflow.log_metric("oof_aucpr", float(oof_ap))
        mlflow.log_metric("oof_best_f1", float(np.max(f1s)))
        mlflow.log_metric("oof_best_thr", best_thr)

        # ROC figure
        fig = plt.figure()
        plt.plot(fprs, tprs); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("OOF ROC")
        local_fig = _report_path("figures", "oof_roc.png")
        fig.savefig(local_fig, bbox_inches="tight")
        mlflow.log_figure(fig, "figures/oof_roc.png")
        plt.close(fig)

        # PR figure
        fig = plt.figure()
        plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("OOF PR")
        local_fig = _report_path("figures", "oof_pr.png")
        fig.savefig(local_fig, bbox_inches="tight")
        mlflow.log_figure(fig, "figures/oof_pr.png")
        plt.close(fig)

        # Save OOF CSV locally and log as artifact
        oof_csv = _report_path("oof", "oof_predictions.csv")
        pd.DataFrame({"oof_pred": oof, "target": y}).to_csv(oof_csv, index=False)
        mlflow.log_artifact(oof_csv, artifact_path="oof")

    return oof

def log_feature_importance(model, X):
    fi = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)

    # Save CSV locally and to MLflow
    fi_csv = _report_path("importance", "feature_importance.csv")
    fi.to_csv(fi_csv, index=False)
    mlflow.log_artifact(fi_csv, artifact_path="importance")

    # Top 30 gain barh
    top = fi.head(30)
    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.xlabel("Gain"); plt.title("Top 30 Feature Importance (gain)")
    plt.tight_layout()

    local_fig = _report_path("figures", "feature_importance_gain.png")
    plt.savefig(local_fig, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), "figures/feature_importance_gain.png")
    plt.close()

def log_shap_plots(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    shap_values_to_use = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values

    # Summary
    shap.summary_plot(shap_values_to_use, X_sample, show=False)
    plt.title("SHAP Summary")
    local_fig = _report_path("figures", "shap_summary.png")
    plt.savefig(local_fig, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), "figures/shap_summary.png")
    plt.close()

    # Dependence plots on top features
    importances = model.feature_importance(importance_type="gain")
    features = np.array(model.feature_name())
    top_features = features[np.argsort(importances)[::-1][:4]]
    for f in top_features:
        shap.dependence_plot(f, shap_values_to_use, X_sample, show=False)
        safe = str(f).replace(os.sep, "_")
        local_fig = _report_path("figures", f"shap_dependence_{safe}.png")
        plt.savefig(local_fig, bbox_inches="tight")
        mlflow.log_figure(plt.gcf(), f"figures/shap_dependence_{safe}.png")
        plt.close()

def log_data_profile(X, y):
    # Summary
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(np.mean(y)),
        "n_categorical": int(len(X.select_dtypes(include=['category']).columns)),
        "n_numeric": int(len(X.select_dtypes(include=[np.number]).columns)),
    }
    summary_csv = _report_path("data_profile", "summary.csv")
    pd.DataFrame([meta]).to_csv(summary_csv, index=False)
    mlflow.log_artifact(summary_csv)

    # Missingness
    miss_csv = _report_path("data_profile", "missingness.csv")
    miss = X.isna().mean().sort_values(ascending=False)
    miss.to_csv(miss_csv, header=["missing_rate"])
    mlflow.log_artifact(miss_csv)

def log_optuna_study(study):
    log_study_figures(study, prefix="figures/optuna_lgbm")
    trials_csv = _report_path("optuna", "lgbm_trials.csv")
    export_trials_csv(study, path=trials_csv)

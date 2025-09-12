from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import mlflow
import xgboost as xgb
import os
from credit_risk.utils.optuna_common import log_study_figures, export_trials_csv

# Reporting paths 
report_base = os.path.join("reports", "xgboost")

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _report_path(*parts: str) -> str:
    path = os.path.join(report_base, *parts)
    _ensure_dir(os.path.dirname(path))
    return path

def _safe_name(name: str) -> str:
    # Avoid directory traversal or separators in filenames
    return str(name).replace(os.sep, "_").replace("/", "_")

def xgb_log_cv_curve(cv_df: pd.DataFrame, name_prefix: str = "cv", metric_key: str | None = None):
    if metric_key is None:
        for k in ["test-auc-mean","test-aucpr-mean","validation-auc-mean","validation-aucpr-mean"]:
            if k in cv_df.columns:
                metric_key = k
                break
        if metric_key is None:
            raise KeyError(f"No known metric column in CV results: {list(cv_df.columns)}")
    std_key = metric_key.replace("-mean","-std") if metric_key.endswith("-mean") else None
    ys = cv_df[metric_key].astype(float).values
    xs = np.arange(1, len(ys)+1)
    plt.figure()
    plt.plot(xs, ys, label=metric_key)
    if std_key and std_key in cv_df.columns:
        stds = cv_df[std_key].astype(float).values
        plt.fill_between(xs, ys-stds, ys+stds, alpha=0.2, label=std_key)
    plt.xlabel("Boosting round"); plt.ylabel(metric_key); plt.title(f"{name_prefix} {metric_key}")
    plt.legend()

    # Save locally and to MLflow
    local_path = _report_path("figures", f"{name_prefix}_cv.png")
    plt.savefig(local_path, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), f"figures/{name_prefix}_cv.png")
    plt.close()

def xgb_train_oof_and_log(params: dict, X: pd.DataFrame, y: np.ndarray, folds: list[tuple[np.ndarray,np.ndarray]]):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score
    oof = np.zeros(len(y), dtype=float)
    with mlflow.start_run(run_name="oof_evaluation_xgb", nested=True):
        for fi,(trn_idx,val_idx) in enumerate(folds):
            dtr = xgb.DMatrix(X.iloc[trn_idx].values, label=y[trn_idx], feature_names=X.columns.tolist())
            dvl = xgb.DMatrix(X.iloc[val_idx].values, label=y[val_idx], feature_names=X.columns.tolist())
            booster = xgb.train(params=params, dtrain=dtr, num_boost_round=5000,
                                evals=[(dvl,"valid")], early_stopping_rounds=200, verbose_eval=False)
            preds = booster.predict(dvl, iteration_range=(0, booster.best_iteration+1))
            oof[val_idx] = preds
            if len(np.unique(y[val_idx])) > 1:
                fpr,tpr,_ = roc_curve(y[val_idx], preds)
                pr,rc,_ = precision_recall_curve(y[val_idx], preds)

                mlflow.log_metric(f"fold{fi}_auc", float(roc_auc_score(y[val_idx], preds)))
                mlflow.log_metric(f"fold{fi}_prauc", float(auc(rc, pr)))

                fig = plt.figure(); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"Fold {fi} ROC")
                local_fig = _report_path("figures", f"fold_{fi}_roc.png")
                fig.savefig(local_fig, bbox_inches="tight")
                mlflow.log_figure(fig, f"figures/fold_{fi}_roc.png"); plt.close(fig)

                fig = plt.figure(); plt.plot(rc,pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Fold {fi} PR")
                local_fig = _report_path("figures", f"fold_{fi}_pr.png")
                fig.savefig(local_fig, bbox_inches="tight")
                mlflow.log_figure(fig, f"figures/fold_{fi}_pr.png"); plt.close(fig)

        if len(np.unique(y)) > 1:
            fpr,tpr,_ = roc_curve(y, oof)
            pr,rc,thr = precision_recall_curve(y, oof)
            f1s = [f1_score(y, (oof>=t).astype(int)) for t in thr[:-1]] if len(thr)>1 else [0.0]

            mlflow.log_metric("oof_auc", float(roc_auc_score(y, oof)))
            mlflow.log_metric("oof_prauc", float(auc(rc, pr)))
            mlflow.log_metric("oof_best_f1", float(np.max(f1s)))
            mlflow.log_metric("oof_best_thr", float(thr[int(np.argmax(f1s))] if len(thr)>1 else 0.5))

            fig = plt.figure(); plt.plot(fpr,tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("OOF ROC")
            local_fig = _report_path("figures", "oof_roc.png")
            fig.savefig(local_fig, bbox_inches="tight")
            mlflow.log_figure(fig, "figures/oof_roc.png"); plt.close(fig)

            fig = plt.figure(); plt.plot(rc,pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("OOF PR")
            local_fig = _report_path("figures", "oof_pr.png")
            fig.savefig(local_fig, bbox_inches="tight")
            mlflow.log_figure(fig, "figures/oof_pr.png"); plt.close(fig)

        # Save OOF CSV locally and log to MLflow
        oof_csv = _report_path("oof", "oof_predictions_xgb.csv")
        pd.DataFrame({"oof_pred": oof, "target": y}).to_csv(oof_csv, index=False)
        mlflow.log_artifact(oof_csv, artifact_path="oof")

    return oof

def xgb_log_feature_importance(booster: xgb.Booster, feature_names: list[str]):
    gain = booster.get_score(importance_type="gain")
    split = booster.get_score(importance_type="weight")
    feats = list(set(feature_names) | set(gain.keys()) | set(split.keys()))
    fi = pd.DataFrame({
        "feature": feats,
        "gain": [gain.get(f,0.0) for f in feats],
        "split": [split.get(f,0.0) for f in feats]
    }).sort_values("gain", ascending=False)

    # CSV locally + MLflow
    fi_csv = _report_path("importance", "feature_importance_xgb.csv")
    fi.to_csv(fi_csv, index=False)
    mlflow.log_artifact(fi_csv, artifact_path="importance")

    # Plot top 30
    top = fi.head(30)
    plt.figure(figsize=(8,10)); plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.xlabel("Gain"); plt.title("Top 30 Feature Importance (XGB Gain)"); plt.tight_layout()

    local_fig = _report_path("figures", "feature_importance_gain_xgb.png")
    plt.savefig(local_fig, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), "figures/feature_importance_gain_xgb.png")
    plt.close()

def xgb_log_shap_plots(booster: xgb.Booster, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_sample, check_additivity=False)

    # Summary
    shap.summary_plot(sv, X_sample, show=False); plt.title("SHAP Summary (XGBoost)")
    local_fig = _report_path("figures", "shap_summary_xgb.png")
    plt.savefig(local_fig, bbox_inches="tight")
    mlflow.log_figure(plt.gcf(), "figures/shap_summary_xgb.png"); plt.close()

    # Dependence plots on top features
    gain = booster.get_score(importance_type="gain")
    top = [f for f,_ in sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    for f in top:
        shap.dependence_plot(f, sv, X_sample, show=False)
        safe = _safe_name(f)
        local_fig = _report_path("figures", f"shap_dependence_xgb_{safe}.png")
        plt.savefig(local_fig, bbox_inches="tight")
        mlflow.log_figure(plt.gcf(), f"figures/shap_dependence_xgb_{safe}.png"); plt.close()

def log_data_profile(X: pd.DataFrame, y: np.ndarray):
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(np.mean(y)),
        "n_numeric": int(len(X.select_dtypes(include=[np.number]).columns)),
    }
    summary_csv = _report_path("data_profile", "summary.csv")
    pd.DataFrame([meta]).to_csv(summary_csv, index=False)
    mlflow.log_artifact(summary_csv)

def log_optuna_study(study):
    log_study_figures(study, prefix="figures/optuna_xgb")
    trials_csv = _report_path("optuna", "xgb_trials.csv")
    export_trials_csv(study, path=trials_csv)

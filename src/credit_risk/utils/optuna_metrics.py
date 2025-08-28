import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import mlflow
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from optuna.visualization.matplotlib import (
    plot_optimization_history, plot_param_importances,
    plot_parallel_coordinate, plot_slice, plot_contour
)

def log_cv_curve(cv_res, name_prefix="cv"):
    aucs = np.asarray(cv_res["valid auc-mean"], dtype=float)
    stds = np.asarray(cv_res.get("valid auc-stdv", np.zeros_like(aucs)), dtype=float)

    xs = np.arange(1, len(aucs) + 1, dtype=int)

    plt.figure()
    plt.plot(xs, aucs, label="AUC (mean)")

    if stds.shape == aucs.shape and np.all(np.isfinite(stds)):
        lower = aucs - stds
        upper = aucs + stds
        plt.fill_between(xs, lower, upper, alpha=0.2, label="±1 std")

    plt.xlabel("Boosting round")
    plt.ylabel("AUC")
    plt.title(f"{name_prefix} AUC over iterations")
    plt.legend()
    mlflow.log_figure(plt.gcf(), f"figures/{name_prefix}_auc_curve.png")
    plt.close()


def train_oof_and_log(best_params, X, y, folds, cat_cols):
    oof = np.zeros(len(y), dtype=float)
    fold_metrics = []

    with mlflow.start_run(run_name="oof_evaluation", nested=True):
        for fi, (trn_idx, val_idx) in enumerate(folds):
            dtr = lgb.Dataset(X.iloc[trn_idx], label=y[trn_idx], categorical_feature=cat_cols, free_raw_data=False)
            dvl = lgb.Dataset(X.iloc[val_idx], label=y[val_idx], categorical_feature=cat_cols, free_raw_data=False)

            booster = lgb.train(best_params, dtr, num_boost_round=5000,
                                valid_sets=[dvl], valid_names=["valid"],
                                callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
            preds = booster.predict(X.iloc[val_idx], num_iteration=booster.best_iteration)
            oof[val_idx] = preds

            fpr, tpr, _ = roc_curve(y[val_idx], preds)
            rocAUC = roc_auc_score(y[val_idx], preds)
            precision, recall, _ = precision_recall_curve(y[val_idx], preds)
            prAUC = auc(recall, precision)

            mlflow.log_metric(f"fold{fi}_auc", float(rocAUC))
            mlflow.log_metric(f"fold{fi}_prauc", float(prAUC))
            fold_metrics.append((rocAUC, prAUC))

        # Aggregate OOF metrics
        oof_auc = roc_auc_score(y, oof)
        precision, recall, thr_pr = precision_recall_curve(y, oof)
        prauc = auc(recall, precision)

        # Choose a threshold that maximizes F1 (optional)
        fprs, tprs, thr_roc = roc_curve(y, oof)
        f1s = [f1_score(y, (oof >= t).astype(int)) for t in thr_pr[:-1]]
        best_idx = int(np.argmax(f1s))
        best_thr = float(thr_pr[best_idx])

        mlflow.log_metric("oof_auc", float(oof_auc))
        mlflow.log_metric("oof_prauc", float(prauc))
        mlflow.log_metric("oof_best_f1", float(f1s[best_idx]))
        mlflow.log_metric("oof_best_thr", best_thr)

        # Log ROC/PR curves as figures
        plt.figure(); plt.plot(fprs, tprs); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("OOF ROC"); 
        mlflow.log_figure(plt.gcf(), "figures/oof_roc.png"); plt.close()

        plt.figure(); plt.plot(recall, precision); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("OOF PR");
        mlflow.log_figure(plt.gcf(), "figures/oof_pr.png"); plt.close()

        # Save OOF preds for later analysis
        oof_df = pd.DataFrame({ "oof_pred": oof, "target": y })
        oof_csv = "oof_predictions.csv"
        oof_df.to_csv(oof_csv, index=False)
        mlflow.log_artifact(oof_csv, artifact_path="oof")

    return oof


def log_feature_importance(model, X):
    importances_gain = model.feature_importance(importance_type="gain")
    importances_split = model.feature_importance(importance_type="split")
    fi = pd.DataFrame({
        "feature": model.feature_name(),
        "gain": importances_gain,
        "split": importances_split
    }).sort_values("gain", ascending=False)
    fi_csv = "feature_importance.csv"
    fi.to_csv(fi_csv, index=False)
    mlflow.log_artifact(fi_csv, artifact_path="importance")

    # Simple bar plot (top 30)
    top = fi.head(30)
    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"][::-1], top["gain"][::-1])
    plt.xlabel("Gain")
    plt.title("Top 30 Feature Importance (gain)")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "figures/feature_importance_gain.png")
    plt.close()


def log_shap_plots(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # LightGBM binary can return list [class0, class1]; prefer positive class
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_use = shap_values[1]
    else:
        shap_values_to_use = shap_values

    shap.summary_plot(shap_values_to_use, X_sample, show=False)
    plt.title("SHAP Summary")
    mlflow.log_figure(plt.gcf(), "figures/shap_summary.png")
    plt.close()

    importances = model.feature_importance(importance_type="gain")
    features = np.array(model.feature_name())
    top_features = features[np.argsort(importances)[::-1][:4]]
    for f in top_features:
        shap.dependence_plot(f, shap_values_to_use, X_sample, show=False)
        mlflow.log_figure(plt.gcf(), f"figures/shap_dependence_{f}.png")
        plt.close()


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_data_profile(X, y):
    _ensure_dir("data_profile")
    meta = {
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "positive_rate": float(np.mean(y)),
        "n_categorical": int(len(X.select_dtypes(include=['category']).columns)),
        "n_numeric": int(len(X.select_dtypes(include=[np.number]).columns)),
    }
    pd.DataFrame([meta]).to_csv("data_profile/summary.csv", index=False)
    mlflow.log_artifact("data_profile/summary.csv")

    miss = X.isna().mean().sort_values(ascending=False)
    miss.to_csv("data_profile/missingness.csv", header=["missing_rate"])
    mlflow.log_artifact("data_profile/missingness.csv")

def log_optuna_study(study):
    figs = [
        ("figures/optuna/opt_history.png", plot_optimization_history(study)),
        ("figures/optuna/param_importances.png", plot_param_importances(study)),
        ("figures/optuna/parallel_coord.png", plot_parallel_coordinate(study)),
        ("figures/optuna/slice.png", plot_slice(study)),
        ("figures/optuna/contour.png", plot_contour(study)),
    ]
    for path, fig in figs:
        mlflow.log_figure(fig, path)
        plt.close(fig)
    # table of trials
    df_trials = study.trials_dataframe(attrs=(
        "number","value","state","params","user_attrs","system_attrs",
        "datetime_start","datetime_complete","duration"
    ))
    _ensure_dir("optuna")
    out = "optuna/trials.csv"
    df_trials.to_csv(out, index=False)
    mlflow.log_artifact(out)
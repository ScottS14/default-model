import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import average_precision_score

import mlflow
import mlflow.lightgbm as mllgb
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna_integration import LightGBMPruningCallback

from credit_risk.utils.optuna_metrics_lgbm import (
    log_cv_curve,
    train_oof_and_log,
    log_feature_importance,
    log_shap_plots,
    log_data_profile,
    log_optuna_study,
)

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("default-model-optuna")

# Custom eval: Average Precision (AUC-PR)
def lgbm_average_precision(preds, train_data):
    y_true = train_data.get_label()
    ap = average_precision_score(y_true, preds)
    # name, value, is_higher_better
    return "average_precision", ap, True

# Data
df = pd.read_parquet("data/processed/train_with_folds_lgbm.parquet")

ID = "SK_ID_CURR"
TARGET = "TARGET"

X = df.drop(columns=[TARGET, ID, "fold"], errors="ignore")
y = df[TARGET].values

cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

if "fold" not in df.columns:
    raise KeyError("Expected a 'fold' column with precomputed fold IDs.")

fold_vals = df["fold"].values
if np.any(fold_vals < 0):
    valid_mask = fold_vals >= 0
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    fold_vals = fold_vals[valid_mask]

unique_folds = np.sort(np.unique(fold_vals))
folds = []
for f in unique_folds:
    val_idx = np.where(fold_vals == f)[0]
    trn_idx = np.where(fold_vals != f)[0]
    folds.append((trn_idx, val_idx))

dtrain = lgb.Dataset(X, label=y, categorical_feature=cat_cols, free_raw_data=False)

# Imbalance helper (for scale_pos_weight search range)
pos = float(y.sum())
neg = float(len(y) - pos)
base_spw = max(neg / max(pos, 1.0), 1.0)

# Optuna objective (optimize AUC-PR)
def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary",
        "metric": "None",   # custom AP metric via feval
        "verbosity": -1,
        "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "goss"]),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 16),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 2.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "is_unbalance": False,
        "scale_pos_weight": trial.suggest_float(name="scale_pos_weight", 
                                                low=0.5 * base_spw, 
                                                high=20.0 * base_spw, log=True),
        "num_threads": os.cpu_count(),
        "seed": 42,
    }

    if params["boosting_type"] == "goss":
        params["bagging_fraction"] = 1.0
        params["bagging_freq"] = 0

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            **params,
            "n_features": X.shape[1],
            "n_rows": X.shape[0],
            "cv_folds": int(unique_folds.size),
            "cv_custom_folds": True,
            "primary_metric": "average_precision",
            "base_scale_pos_weight": base_spw,
            "prevalence": float(y.mean()),
        })

        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=False),
            LightGBMPruningCallback(trial, "average_precision"),
        ]

        cv_res = lgb.cv(
            params,
            dtrain,
            folds=folds,
            num_boost_round=5000,
            feval=lgbm_average_precision,
            callbacks=callbacks,
            return_cvbooster=False,
        )

        # Log learning curve
        log_cv_curve(cv_res, name_prefix=f"trial_{trial.number}")

        # Extract best AP across rounds
        ap_mean_key = "valid average_precision-mean"
        ap_std_key  = "valid average_precision-stdv"
        if ap_mean_key not in cv_res:
            raise KeyError(f"Expected '{ap_mean_key}' in CV results. Keys: {list(cv_res.keys())}")

        aps = cv_res[ap_mean_key]
        stds = cv_res.get(ap_std_key, [0.0] * len(aps))

        best_iter = int(np.argmax(aps)) + 1
        best_ap = float(aps[best_iter - 1])
        best_std = float(stds[best_iter - 1])

        # Log trial metrics
        mlflow.log_metric("aucpr_mean", best_ap)
        mlflow.log_metric("aucpr_std", best_std)
        mlflow.log_metric("best_iter", best_iter)

        # Report to Optuna for pruning
        for step, v in enumerate(aps):
            trial.report(float(v), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_ap

# Run the study
study_name = "lgbm_optuna_aucpr_4"
storage = "sqlite:///optuna_study_lightgbm.db"

sampler = TPESampler(seed=42, multivariate=True, group=True)
pruner = HyperbandPruner(min_resource=100, max_resource=5000, reduction_factor=3)

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    sampler=sampler,
    pruner=pruner
)

with mlflow.start_run(run_name="lgbm_cv_optuna_study"):
    mlflow.log_param("optuna_sampler", "TPE")
    mlflow.log_param("optuna_pruner", "Hyperband")
    mlflow.log_param("cv_custom_folds", True)
    mlflow.log_param("n_folds", int(unique_folds.size))
    mlflow.set_tag("dataset", "train_with_folds_lgbm.parquet")
    mlflow.set_tag("task", "binary_classification")
    mlflow.set_tag("framework", "lightgbm")
    mlflow.set_tag("primary_metric", "average_precision")

    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # Log Optuna visuals & trials table
    log_optuna_study(study)

    best_trial = study.best_trial
    best_params = best_trial.params

    if best_params.get("boosting_type") == "goss":
        best_params["bagging_fraction"] = 1.0
        best_params["bagging_freq"] = 0

    best_params.update({
        "objective": "binary",
        "metric": "None",        
        "num_threads": os.cpu_count(),
        "seed": 42,
        "verbosity": -1,
    })

    # Re-run CV with best params using AP
    cv_res = lgb.cv(
        best_params,
        dtrain,
        folds=folds,
        num_boost_round=5000,
        feval=lgbm_average_precision,
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)],
        return_cvbooster=False,
    )
    log_cv_curve(cv_res, name_prefix="final_params")

    ap_mean_key = "valid average_precision-mean"
    aps = cv_res[ap_mean_key]
    best_iter = int(np.argmax(aps)) + 1
    best_cv_aucpr = float(aps[best_iter - 1])

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_aucpr", best_cv_aucpr)
    mlflow.log_metric("best_iter", best_iter)

    # Train final model at best_iter and log it
    final_model = lgb.train(
        best_params,
        dtrain,
        num_boost_round=best_iter,
        feval=lgbm_average_precision
    )

    # OOF + importance + SHAP + data profile via utils
    oof = train_oof_and_log(best_params, X, y, folds, cat_cols) 
    log_feature_importance(final_model, X)

    try:
        X_sample = X.sample(min(10000, len(X)), random_state=42)
        log_shap_plots(final_model, X_sample)
    except Exception as e:
        mlflow.log_text(str(e), "figures/shap_error.txt")

    log_data_profile(X, y)

    # Save Optuna best summary file
    try:
        study_summary = {
            "best_trial_number": best_trial.number,
            "best_value_aucpr": best_trial.value,
            "best_params": best_params,
            "prevalence": float(y.mean()),
        }
        import json, tempfile
        tmp_path = os.path.join(tempfile.gettempdir(), "optuna_best.json")
        with open(tmp_path, "w") as f:
            json.dump(study_summary, f, indent=2)
        mlflow.log_artifact(tmp_path, artifact_path="optuna")
    except Exception:
        pass

    # Log model
    mllgb.log_model(final_model, artifact_path="model")

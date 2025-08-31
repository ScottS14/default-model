import os
import json
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.sparse as sp
import mlflow
import mlflow.xgboost as mlxgb
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from optuna_integration.xgboost import XGBoostPruningCallback

from credit_risk.utils.optuna_metrics_xgb import (
    xgb_log_cv_curve,
    xgb_train_oof_and_log,
    xgb_log_feature_importance,
    xgb_log_shap_plots,
    log_data_profile,
    log_optuna_study,
)

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("default-model-optuna")

ID = "SK_ID_CURR"
TARGET = "TARGET"

df = pd.read_parquet("data/processed/train_features_xgb.parquet")
fold_id = pd.read_parquet("data/processed/folds_by_sk_id.parquet")

if ID not in fold_id.columns and "row_id" in fold_id.columns:
    fold_id[ID] = fold_id["row_id"]

df_with_folds = fold_id.merge(df, how="left", on=ID)

if "fold" not in df_with_folds.columns:
    raise KeyError("Expected a 'fold' column with precomputed fold IDs in the merged dataframe.")

X = df_with_folds.drop(columns=[TARGET, ID, "fold"], errors="ignore")
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(np.uint8)
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.astype(np.float32)
y = df_with_folds[TARGET].values.astype(np.float32)

fold_vals = df_with_folds["fold"].values
if np.any(fold_vals < 0):
    m = fold_vals >= 0
    X = X.loc[m].reset_index(drop=True)
    y = y[m]
    fold_vals = fold_vals[m]

unique_folds = np.sort(np.unique(fold_vals))
folds = []
for f in unique_folds:
    val_idx = np.where(fold_vals == f)[0]
    trn_idx = np.where(fold_vals != f)[0]
    folds.append((trn_idx, val_idx))

X_csr = sp.csr_matrix(X.values)
feature_names = X.columns.tolist()
dtrain = xgb.DMatrix(X_csr, label=y, feature_names=feature_names)

def _detect_cv_keys(trial, params):
    probe = xgb.cv(params=params, dtrain=dtrain, folds=folds, num_boost_round=1, verbose_eval=False, seed=42)
    cols = list(probe.columns)
    print(f"[trial {trial.number}] xgb.cv columns: {cols}")
    try:
        mlflow.log_text(str(cols), f"trial_{trial.number}_cv_columns.txt")
    except Exception:
        pass
    for base in ["test-auc", "validation-auc", "test-aucpr", "validation-aucpr"]:
        mean_k, std_k = f"{base}-mean", f"{base}-std"
        if mean_k in probe.columns:
            return mean_k, (std_k if std_k in probe.columns else None), base
    for c in cols:
        if ("test" in c or "validation" in c) and ("auc" in c):
            base = c.split("-mean")[0].split("-std")[0]
            stdk = c.replace("-mean", "-std") if c.endswith("-mean") else None
            return c, stdk, base
    return None, None, None

def objective(trial: optuna.Trial) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": trial.suggest_categorical("eval_metric", ["auc", "aucpr"]),
        "tree_method": "hist",
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 50.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "eta": trial.suggest_float("eta", 0.005, 0.1, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0, log=True),
        "seed": 42,
    }

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            **params,
            "n_features": X.shape[1],
            "n_rows": X.shape[0],
            "cv_folds": int(unique_folds.size),
            "cv_custom_folds": True,
            "dtype": "float32",
        })

        metric_key, std_key, prune_base = _detect_cv_keys(trial, params)
        if metric_key is None:
            mlflow.log_text("Could not detect CV metric key.", f"errors/trial_{trial.number}_metric_key.txt")
            raise optuna.TrialPruned()

        callbacks = []
        try:
            callbacks = [XGBoostPruningCallback(trial, prune_base)]
        except Exception as e:
            mlflow.log_text(str(e), f"errors/trial_{trial.number}_prune_setup.txt")
            callbacks = []

        try:
            cv_res = xgb.cv(
                params=params,
                dtrain=dtrain,
                folds=folds,
                num_boost_round=5000,
                early_stopping_rounds=200,
                verbose_eval=False,
                callbacks=callbacks,
                seed=42,
            )
        except KeyError as e:
            mlflow.log_text(f"Pruning key mismatch: {e}", f"errors/trial_{trial.number}_prune_keyerror.txt")
            cv_res = xgb.cv(
                params=params,
                dtrain=dtrain,
                folds=folds,
                num_boost_round=5000,
                early_stopping_rounds=200,
                verbose_eval=False,
                seed=42,
            )

        xgb_log_cv_curve(cv_res, name_prefix=f"trial_{trial.number}", metric_key=metric_key)

        aucs = cv_res[metric_key].astype(float)
        stds = (cv_res[std_key].astype(float) if std_key and std_key in cv_res.columns
                else pd.Series(np.zeros(len(aucs)), index=aucs.index))

        best_idx = int(aucs.idxmax())
        best_iter = best_idx + 1
        best_auc = float(aucs.iloc[best_idx])
        best_std = float(stds.iloc[best_idx])

        mlflow.log_metric("auc_mean", best_auc)
        mlflow.log_metric("auc_std", best_std)
        mlflow.log_metric("best_iter", best_iter)

        for i, v in enumerate(aucs, start=1):
            trial.report(float(v), step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_auc

study_name = "xgb_optuna_auc_2"
storage = "sqlite:///optuna_study_xgboost.db"

sampler = TPESampler(seed=42, multivariate=True, group=True)
pruner = HyperbandPruner(min_resource=100, max_resource=5000, reduction_factor=3)

study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    sampler=sampler,
    pruner=pruner,
)

with mlflow.start_run(run_name="xgboost_cv_optuna_study"):
    mlflow.log_param("optuna_sampler", "TPE")
    mlflow.log_param("optuna_pruner", "Hyperband")
    mlflow.log_param("cv_custom_folds", True)
    mlflow.log_param("n_folds", int(unique_folds.size))
    mlflow.set_tag("dataset", "train_features_xgb.parquet")
    mlflow.set_tag("task", "binary_classification")
    mlflow.set_tag("framework", "xgboost")

    study.optimize(objective, n_trials=50, show_progress_bar=False, catch=(Exception,))

    log_optuna_study(study)

    best_trial = study.best_trial
    best_params = best_trial.params.copy()
    best_params.update({
        "objective": "binary:logistic",
        "eval_metric": best_params.get("eval_metric", "auc"),
        "tree_method": "hist",
        "seed": 42,
    })

    mk, sk, _ = _detect_cv_keys(best_trial, best_params)
    cv_res = xgb.cv(
        params=best_params,
        dtrain=dtrain,
        folds=folds,
        num_boost_round=5000,
        early_stopping_rounds=200,
        verbose_eval=False,
        seed=42,
    )
    if mk is None or mk not in cv_res.columns:
        mlflow.log_text(str(list(cv_res.columns)), "errors/final_cv_cols.txt")
        raise RuntimeError("Could not locate CV metric column.")
    xgb_log_cv_curve(cv_res, name_prefix="final_params", metric_key=mk)

    aucs = cv_res[mk].astype(float)
    best_idx = int(aucs.idxmax())
    best_iter = best_idx + 1
    best_cv_auc = float(aucs.iloc[best_idx])

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_auc", best_cv_auc)
    mlflow.log_metric("best_iter", best_iter)

    final_model = xgb.train(params=best_params, dtrain=dtrain, num_boost_round=best_iter)

    xgb_train_oof_and_log(best_params, X, y, folds)
    xgb_log_feature_importance(final_model, feature_names)

    try:
        X_sample = X.sample(min(10000, len(X)), random_state=42)
        xgb_log_shap_plots(final_model, X_sample)
    except Exception as e:
        mlflow.log_text(str(e), "figures/shap_error.txt")

    log_data_profile(X, y)

    try:
        summary = {"best_trial_number": best_trial.number, "best_value": best_trial.value, "best_params": best_params}
        tmp_path = os.path.join(tempfile.gettempdir(), "optuna_best_xgb.json")
        with open(tmp_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(tmp_path, artifact_path="optuna")
    except Exception:
        pass

    mlxgb.log_model(final_model, artifact_path="model")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import mlflow
import mlflow.xgboost as mlfxgb
import xgboost as xgb

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("default-model-baseline")
mlfxgb.autolog()

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

for col in X.columns:
    if X[col].dtype == bool:
        X[col] = X[col].astype(np.uint8)
    if not np.issubdtype(X[col].dtype, np.number):
        X[col] = pd.to_numeric(X[col], errors="coerce")
X = X.astype(np.float32)

y = df_with_folds[TARGET].values.astype(np.float32)

# Handle invalid folds (e.g., -1)
fold_vals = df_with_folds["fold"].values
if np.any(fold_vals < 0):
    valid_mask = fold_vals >= 0
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y[valid_mask]
    fold_vals = fold_vals[valid_mask]

# Custom folds list of (train_idx, valid_idx)
unique_folds = np.sort(np.unique(fold_vals))
folds = []
for f in unique_folds:
    val_idx = np.where(fold_vals == f)[0]
    trn_idx = np.where(fold_vals != f)[0]
    folds.append((trn_idx, val_idx))

X_csr = sp.csr_matrix(X.values)
dtrain = xgb.DMatrix(X_csr, label=y, feature_names=X.columns.tolist())

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "seed": 42,
}

with mlflow.start_run(run_name="xgboost_cv_baseline_ohe"):
    mlflow.log_params({
        **params,
        "n_features": X.shape[1],
        "n_rows": X.shape[0],
        "cv_folds": int(unique_folds.size),
        "cv_custom_folds": True,
        "ohe_sparse": True,
        "dtype": "float32",
    })

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        folds=folds,
        num_boost_round=5000,
        early_stopping_rounds=100,
        verbose_eval=False,
        seed=42,
    )

    aucs = cv_results["test-auc-mean"]
    stds = cv_results["test-auc-std"]
    best_iter = int(aucs.idxmax()) + 1
    best_auc = float(aucs.iloc[best_iter - 1])
    best_std = float(stds.iloc[best_iter - 1])

    mlflow.log_metric("auc_mean", best_auc)
    mlflow.log_metric("auc_std", best_std)
    mlflow.log_metric("best_iter", best_iter)

    
    final_model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=best_iter,
    )

    mlfxgb.log_model(final_model, artifact_path="model")

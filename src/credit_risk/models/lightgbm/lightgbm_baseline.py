import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm as mllgb

mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("default-model-baseline")
mlflow.autolog()

df = pd.read_parquet("data/processed/train_with_folds_lgbm.parquet")

ID = "SK_ID_CURR"
TARGET = "TARGET"

#  Build X/y and remember folds 
X = df.drop(columns=[TARGET, ID, "fold"], errors="ignore")
y = df[TARGET].values

cat_cols = X.select_dtypes(include=["category"]).columns.tolist()

if "fold" not in df.columns:
    raise KeyError("Expected a 'fold' column with precomputed fold IDs.")

fold_vals = df["fold"].values
if np.any(fold_vals < 0):
    # ignore rows without a valid fold assignment (e.g., -1)
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

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "is_unbalance": True,
    "seed": 42,
}

with mlflow.start_run(run_name="lgbm_cv_baseline"):
    mlflow.log_params({
        **params,
        "n_features": X.shape[1],
        "n_rows": X.shape[0],
        "cv_folds": int(unique_folds.size),
        "cv_custom_folds": True,
    })

    # Custom flods
    cv = lgb.cv(
        params,
        dtrain,
        folds=folds,
        num_boost_round=5000,
        callbacks=[lgb.early_stopping(stopping_rounds=100)],
    )

    aucs = cv["valid auc-mean"]
    stds = cv["valid auc-stdv"]
    best_iter = int(np.argmax(aucs)) + 1
    best_auc = float(aucs[best_iter - 1])
    best_std = float(stds[best_iter - 1])

    mlflow.log_metric("auc_mean", best_auc)
    mlflow.log_metric("auc_std", best_std)
    mlflow.log_metric("best_iter", best_iter)

    
    final_model = lgb.train(params, dtrain, num_boost_round=best_iter)

    mllgb.log_model(final_model, name="model")

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm as mllgb

# point to the tracking server
mlflow.set_tracking_uri("http://localhost:5500")
mlflow.set_experiment("default-model-baseline")
mlflow.autolog()

load_data = pd.read_parquet('data/processed/train_features_lgbm.parquet')

cols = [col for col in load_data.columns if col != 'TARGET']

X = load_data[cols]
y = load_data['TARGET']

dtrain = lgb.Dataset(X, label=y)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,     
    "num_leaves": 64,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "is_unbalance": True,      
    "seed": 42                 
}

with mlflow.start_run(run_name="lgbm_cv_baseline"):
    mlflow.log_params({
        **params,
        "n_features": X.shape[1],
        "n_rows": X.shape[0],
        "cv_folds": 5
    })

    cv = lgb.cv(
        params,
        dtrain,
        num_boost_round=5000,
        nfold=5,
        stratified=True,
        shuffle=True,
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

    # Train a final model on all data at best_iter and log it
    final_model = lgb.train(params, dtrain, num_boost_round=best_iter)
    mllgb.log_model(final_model, name="model")

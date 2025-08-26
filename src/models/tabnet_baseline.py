import os, numpy as np, pandas as pd, mlflow, torch
from sklearn.metrics import roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier

ID, TARGET = "SK_ID_CURR", "TARGET"
DATA_PATH = "data/processed/train_with_folds_lgbm.parquet"

def encode_cats_no_negatives(Xdf):
    cat_cols = Xdf.select_dtypes(include=["category"]).columns.tolist()
    cat_idxs, cat_dims = [], []
    cols = Xdf.columns.tolist()
    for i, c in enumerate(cols):
        if c in cat_cols:
            codes = Xdf[c].cat.codes.astype("int64")
            # replace -1 (NaN) with new category id
            codes = np.where(codes < 0, len(Xdf[c].cat.categories), codes)
            Xdf[c] = codes
            cat_idxs.append(i)
            cat_dims.append(int(Xdf[c].max()) + 1)
        else:
            if Xdf[c].dtype == bool:
                Xdf[c] = Xdf[c].astype("int64")
            elif not np.issubdtype(Xdf[c].dtype, np.number):
                Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    return Xdf.fillna(0), cat_idxs, cat_dims

def load_data(path):
    df = pd.read_parquet(path)
    if "fold" not in df.columns:
        raise KeyError("Expected a 'fold' column.")
    df = df.loc[df["fold"].values >= 0].reset_index(drop=True)

    Xdf = df.drop(columns=[TARGET, ID, "fold"], errors="ignore").copy()
    y = df[TARGET].astype(np.int64).values
    folds = df["fold"].values

    Xdf, cat_idxs, cat_dims = encode_cats_no_negatives(Xdf)
    X = Xdf.astype(np.float32).values
    return X, y, folds, cat_idxs, cat_dims

def run_cv(X, y, folds, cat_idxs, cat_dims, params, max_epochs=200, patience=20, verbose=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aucs = []
    uniq = np.sort(np.unique(folds))
    last_model = None

    for f in uniq:
        val_idx = np.where(folds == f)[0]
        trn_idx = np.where(folds != f)[0]

        X_tr, y_tr = X[trn_idx], y[trn_idx]
        X_va, y_va = X[val_idx], y[val_idx]

        clf = TabNetClassifier(
            n_d=params["n_d"], n_a=params["n_a"], n_steps=params["n_steps"],
            gamma=params["gamma"], lambda_sparse=params["lambda_sparse"],
            momentum=params["momentum"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params["learning_rate"]),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size":50, "gamma":0.9},
            mask_type="entmax",
            cat_idxs=cat_idxs, cat_dims=cat_dims,
            cat_emb_dim=params["cat_emb_dim"],
            device_name=device,
        )

        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_name=["valid"], eval_metric=["auc"],
            max_epochs=max_epochs, patience=patience,
            batch_size=params["batch_size"], virtual_batch_size=params["virtual_batch_size"],
            num_workers=0, drop_last=False
        )

        y_hat = clf.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, y_hat)
        aucs.append(auc)
        last_model = clf

    return float(np.mean(aucs)), float(np.std(aucs)), last_model

def main(data_path=DATA_PATH, out_dir="./runs/tabnet_cv_baseline"):
    os.makedirs(out_dir, exist_ok=True)

    # MLflow storage
    mlruns_uri = "file:/content/drive/MyDrive/mlruns" if os.path.exists("/content") else "file:./mlruns"
    mlflow.set_tracking_uri(mlruns_uri)
    mlflow.set_experiment("tabnet-baseline")

    X, y, folds, cat_idxs, cat_dims = load_data(data_path)

    params = dict(
        n_d=16, n_a=16, n_steps=3, gamma=1.3,
        lambda_sparse=1e-5, momentum=0.02,
        learning_rate=2e-3, batch_size=1024, virtual_batch_size=128,
        cat_emb_dim=8,
    )

    with mlflow.start_run(run_name="tabnet_cv_baseline"):
        mlflow.log_params({
            **params,
            "n_features": int(X.shape[1]),
            "n_rows": int(X.shape[0]),
            "cv_folds": int(np.unique(folds).size),
            "cv_custom_folds": True,
            "n_cats": len(cat_idxs),
        })

        mean_auc, std_auc, model = run_cv(
            X, y, folds, cat_idxs, cat_dims, params,
            max_epochs=200, patience=20, verbose=0
        )

        mlflow.log_metric("auc_mean", mean_auc)
        mlflow.log_metric("auc_std", std_auc)

        # save model weights + cat spec
        torch.save(model.network.state_dict(), os.path.join(out_dir, "tabnet_state.pt"))
        mlflow.log_artifact(os.path.join(out_dir, "tabnet_state.pt"), artifact_path="model")
        import json
        with open(os.path.join(out_dir, "cat_spec.json"), "w") as f:
            json.dump({"cat_idxs": cat_idxs, "cat_dims": cat_dims}, f)
        mlflow.log_artifact(os.path.join(out_dir, "cat_spec.json"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=DATA_PATH)
    ap.add_argument("--out_dir", type=str, default="./runs/tabnet_cv_baseline")
    args = ap.parse_args()
    main(args.data, args.out_dir)

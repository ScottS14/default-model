from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

try:
    import mlflow
except Exception:
    mlflow = None

def log_calibration_and_brier(y_true: np.ndarray, y_prob: np.ndarray, out_dir: str, tag: str, log_mlflow: bool):
    os.makedirs(out_dir, exist_ok=True)

    brier = float(brier_score_loss(y_true, y_prob))
    with open(os.path.join(out_dir, f"{tag}_brier.txt"), "w") as f:
        f.write(f"{brier:.6f}\n")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    cal_df = pd.DataFrame({"pred_mean": prob_pred, "obs_rate": prob_true})
    cal_df.to_csv(os.path.join(out_dir, f"{tag}_calibration_table.csv"), index=False)

    plt.figure()
    plt.plot([0,1], [0,1], "--", linewidth=1)
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve ({tag})")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{tag}_calibration_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    if log_mlflow and mlflow is not None:
        mlflow.log_metric(f"{tag}_brier", brier)
        mlflow.log_artifact(fig_path, artifact_path="figures")
        mlflow.log_artifact(os.path.join(out_dir, f"{tag}_calibration_table.csv"), artifact_path="figures")
        mlflow.log_artifact(os.path.join(out_dir, f"{tag}_brier.txt"), artifact_path="figures")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-csv", required=True, help="CSV with columns: oof_pred, target")
    p.add_argument("--out-dir", default="reports/figures")
    p.add_argument("--tag", default="oof")
    p.add_argument("--mlflow", action="store_true", help="Also log to MLflow if available")
    args = p.parse_args()

    df = pd.read_csv(args.oof_csv)
    y_prob = df["oof_pred"].astype(float).values
    y_true = df["target"].astype(int).values
    log_calibration_and_brier(y_true, y_prob, args.out_dir, args.tag, args.mlflow)

if __name__ == "__main__":
    main()

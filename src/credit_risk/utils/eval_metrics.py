from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import mlflow

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def log_calibration_and_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    out_dir: str,
    tag: str,
    log_to_mlflow: bool = False
):
    _ensure_dir(out_dir)

    # Brier score
    brier = float(brier_score_loss(y_true, y_prob))
    brier_path = os.path.join(out_dir, f"{tag}_brier.txt")
    with open(brier_path, "w") as f:
        f.write(f"{brier:.6f}\n")

    # Calibration table
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    cal_df = pd.DataFrame({"pred_mean": prob_pred, "obs_rate": prob_true})
    cal_path = os.path.join(out_dir, f"{tag}_calibration_table.csv")
    cal_df.to_csv(cal_path, index=False)

    # Calibration curve plot
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(f"Calibration curve ({tag})")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{tag}_calibration_curve.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    if log_to_mlflow and mlflow is not None:
        try:
            mlflow.log_artifact(brier_path, artifact_path="calibration")
            mlflow.log_artifact(cal_path, artifact_path="calibration")
            mlflow.log_artifact(fig_path, artifact_path="calibration")
        except Exception:
            pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--oof-csv", required=True, help="CSV with columns: oof_pred, target")
    p.add_argument("--model", choices=["lgbm", "xgboost"], default="lgbm",
                  help="Model family to route outputs under reports/<model>/calibration")
    p.add_argument("--out-dir", default=None,
                  help="Optional override for output dir. Default is reports/<model>/calibration")
    p.add_argument("--tag", default="oof")
    p.add_argument("--mlflow", action="store_true", help="Also log artifacts to MLflow under 'calibration/'")
    args = p.parse_args()

    # Hard-coded base
    base = f"reports/{args.model}/calibration"
    out_dir = args.out_dir if args.out_dir else base

    df = pd.read_csv(args.oof_csv)
    y_prob = df["oof_pred"].astype(float).values
    y_true = df["target"].astype(int).values

    log_calibration_and_brier(y_true, y_prob, out_dir, args.tag, log_to_mlflow=args.mlflow)

if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, os, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

def _fit_calibrator(y_true: np.ndarray, y_prob: np.ndarray, method: str):
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(y_prob, y_true)
        return cal, (lambda p: cal.transform(p))
    else:
        raise ValueError("method must be 'isotonic'")

def _plot_calibration(y_true, y_prob, y_prob_cal, out_png: str, title: str):
    plt.figure()
    plt.plot([0,1],[0,1],"--", linewidth=1)
    for preds, lbl in [(y_prob, "raw"), (y_prob_cal, "calibrated")]:
        pt, pp = calibration_curve(y_true, preds, n_bins=10, strategy="quantile")
        plt.plot(pp, pt, marker="o", linewidth=1, label=lbl)
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def fit_from_oof(oof_csv: str, out_calibrator: str, method: str, fig_dir: str):
    os.makedirs(os.path.dirname(out_calibrator) or ".", exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df = pd.read_csv(oof_csv)
    y_true = df["target"].astype(int).values
    y_prob = df["oof_pred"].astype(float).values

    raw_brier = brier_score_loss(y_true, y_prob)
    cal, apply_fn = _fit_calibrator(y_true, y_prob, method)
    y_prob_cal = apply_fn(y_prob)
    cal_brier = brier_score_loss(y_true, y_prob_cal)

    joblib.dump(cal, out_calibrator)

    # plots + summary
    _plot_calibration(y_true, y_prob, y_prob_cal,
                      os.path.join(fig_dir, f"calibration_{method}.png"),
                      f"Calibration ({method})")
    with open(os.path.join(fig_dir, f"brier_{method}.txt"), "w") as f:
        f.write(f"raw_brier={raw_brier:.6f}\ncalibrated_brier={cal_brier:.6f}\n")

    print(f"[{method}] Brier raw={raw_brier:.6f} → calibrated={cal_brier:.6f}")
    print(f"Saved calibrator → {out_calibrator}")

def apply_calibrator(pred_csv_in: str, pred_col: str, calibrator_pkl: str, pred_csv_out: str):
    df = pd.read_csv(pred_csv_in)
    cal = joblib.load(calibrator_pkl)

    def _apply(p):
        return cal.transform(p)
        

    p = df[pred_col].astype(float).values
    df[pred_col + "_calibrated"] = _apply(p)
    df.to_csv(pred_csv_out, index=False)
    print(f"Wrote calibrated preds → {pred_csv_out}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="fit calibrator from OOF")
    f.add_argument("--oof-csv", required=True)
    f.add_argument("--out-calibrator", required=True)
    f.add_argument("--method", choices=["isotonic"], default="isotonic")
    f.add_argument("--fig-dir", default="reports/figures/calibration")

    a = sub.add_parser("apply", help="apply saved calibrator to preds")
    a.add_argument("--pred-csv-in", required=True)
    a.add_argument("--pred-col", default="prediction")
    a.add_argument("--calibrator-pkl", required=True)
    a.add_argument("--pred-csv-out", required=True)

    args = ap.parse_args()
    if args.cmd == "fit":
        fit_from_oof(args.oof_csv, args.out_calibrator, args.method, args.fig_dir)
    else:
        apply_calibrator(args.pred_csv_in, args.pred_col, args.calibrator_pkl, args.pred_csv_out)

if __name__ == "__main__":
    main()

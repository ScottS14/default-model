import joblib
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

df = pd.read_csv("oof/oof_predictions.csv")
y = df["target"].astype(int).values
s = df["oof_pred"].astype(float).values

iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(s, y)

cal_oof = iso.transform(s)
print("Brier (raw):", brier_score_loss(y, s))
print("Brier (iso):", brier_score_loss(y, cal_oof))

joblib.dump(iso, "models/calibration/isotonic.joblib")

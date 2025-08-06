# Credit-Risk Scoring – Home-Credit Default Risk 🚀
*Predict the probability that an applicant will default, using open Home-Credit data.*

---

## 1  Problem statement
Lenders need to decide **“approve or refuse?”** within seconds while keeping default losses low and meeting regulatory explainability standards.

> **Goal:** build a reproducible pipeline that ingests the public *Home-Credit Default Risk* dataset, engineers economically meaningful features, trains baseline models (LightGBM/XGBoost), and surfaces explanations (SHAP) suitable for credit-risk governance docs.

---

## 2  Project roadmap 
| Phase | Status | ETA |
|-------|--------|-----|
| EDA and Feature engineering script (`engineer_credit.py`) | ▢ | Week 1 |
| Baseline LightGBM & XGBoost (CV) | ▢ | Week 2 |
| Hyper-param tuning + calibration | ▢ | Week 3 |
| SHAP explainability | ▢ | Week 4 |
| Docker < 400 MB + v1.0.0 release | ▢ | Week 5 |

*(Full multi-project timeline lives in `/docs/timeline.md`.)*

---

## 3  Dataset 
Home-Credit competition files are **not committed** to Git to keep the repo lean.

```bash
# One-time download (requires kaggle CLI & creds)
bash scripts/fetch_data.sh

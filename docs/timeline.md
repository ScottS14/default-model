# Timeline: Project – Credit-Risk Scoring

**Epic Goal**  
Deliver a fully reproducible credit-risk model for the Home-Credit dataset, published as `v1.0.0` on GitHub with a Zenodo DOI and a < 400 MB Docker image built with `uv`.

| Week | Story | Points | Sub-tasks | Acceptance Criteria |
|------|-------|--------|-----------|---------------------|
| **1** | **Data audit & baseline models** | 8 | **EDA / data-quality**<br>- [ ] Draft `clean_dataframe` with unit tests<br>- [ ] Missingness heatmap + sentinel audit (`eda_overview.ipynb`)<br><br>**Feature scaffolding & first models**<br>- [ ] Draft aggregation functions<br>- [ ] Target-encoding utilities<br>- [ ] Train LightGBM 5-fold CV<br>- [ ] Train XGBoost CV<br>- [ ] Quick TabNet baseline | • `eda_overview.ipynb` saved under `/notebooks/`<br>• `cleaning.py` & tests pass CI<br>• ≥ 10 MLflow runs in **CR-Baseline**<br>• ROC & PR curves saved to `/reports/figures/` |
| **2** | **Hyper-param tuning & interpretability** | 13 | - [ ] Launch Optuna study (50 trials)<br>- [ ] Generate SHAP beeswarm + force plots<br>- [ ] Calibration curve & Brier score<br>- [ ] README update | • Optuna DB committed; README lists top params & metrics<br>• Beeswarm + ≥ 1 force plot<br>• Calibration curve present; **Brier ≤ 0.18** |
| **3** | **Package & publish v1.0.0** | 5 | - [ ] Production Dockerfile (< 400 MB)<br>- [ ] Loom walkthrough (3 min)<br>- [ ] Tag `v1.0.0` & Zenodo DOI<br>- [ ] LinkedIn draft | • `docker run` reproduces results end-to-end<br>• GitHub release shows DOI; Loom link works |

---

### Detailed Breakdown

<details>
<summary>Week&nbsp;1-2 – Baseline classical & deep models</summary>

#### Objectives
1. **Understand & clean** the raw Home-Credit tables so modelling can trust the inputs.  
2. **Establish baselines** for classical (LightGBM/XGBoost) and deep-tabular (TabNet) approaches.

#### Tasks
- **CRS-1-a** `clean_dataframe` implementation + pytest edge cases.  
- **CRS-1-b** `eda_overview.ipynb` with missingness map, target imbalance, sentinel checks.  
- **CRS-1-c** Draft *aggregation functions* in `feature_eng.py`.  
- **CRS-1-d** Add *target-encoding utilities* (fit/transform API).  
- **CRS-1-e** Train **LightGBM** 5-fold CV; log metrics to MLflow.  
- **CRS-1-f** Train **XGBoost** 5-fold CV; mirror LightGBM procedure.  
- **CRS-1-g** Quick **TabNet** sanity run (1 epoch).

</details>

<details>
<summary>Week&nbsp;3-4 – Hyper-param tuning & interpretability</summary>

#### Objectives
Improve the baseline with Optuna HPO; provide model transparency and calibrated probabilities.

#### Tasks
- **CRS-2-a** 50-trial *Optuna study* on LightGBM; commit `optuna_study_lgb.db`.
- **CRS-2-b** 50-trial *Optuna study* on XGBoost; commit `optuna_study_xgb.db`.
- **CRS-2-c** Generate *SHAP beeswarm* and force plots for five customers.
- **CRS-2-d** Plot *probability-calibration curve*; compute *Brier score*.
- **CRS-2-e** Calibrate the results
- **CRS-2-f** Update **README** with Model Zoo table + metrics.

</details>

<details>
<summary>Week&nbsp;4-5 – Package & publish v1.0.0</summary>

#### Objectives
Ship a lean, reproducible artifact and public release.

#### Tasks
- **CRS-3-a** Build production *Dockerfile* (< 400 MB) and push to GHCR.
- **CRS-3-b** Record 3-minute *Loom walkthrough*; store link in `/docs/links.md`.
- **CRS-3-c** Tag **v1.0.0**, archive via Zenodo, embed DOI badge.
- **CRS-3-d** Draft *LinkedIn promo* with hero image (`promo_linkedin.md`).

</details>

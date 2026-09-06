# Corporate Bankruptcy Risk Prediction

An end-to-end ML project that predicts a company's bankruptcy probability from 65 financial ratios and explains *why* using SHAP. Dataset: Taiwan Economic Journal — 25,121 companies, 65 ratios, 1 target (`class`: 1 = bankrupt).

---

## Project structure

```
bankruptcy-prediction/
├── app/
│   ├── inference.py        # raw ratios -> preprocessed features -> prediction + SHAP factors
│   └── dashboard.py        # Streamlit UI (manual entry / CSV upload)
├── scripts/                # numbered, run-in-order pipeline (the real logic lives here)
├── src/                     # package skeleton — currently empty (see Known issues)
├── data/raw/ , data/processed/
├── models/                  # xgboost_baseline.pkl, xgboost_tuned.pkl, catboost_baseline.pkl
├── reports/                 # metrics, SHAP importances, figures
├── tests/                   # test_pipeline.py — currently empty
├── configs/config.yaml      # currently empty
└── requirements.txt
```

---

## What we did, and why

### 1. Data cleaning (`02_data_cleaning.py`)
**What:** converted raw text columns to numeric, checked duplicates, quantified missing values, and ran a chi-square test per column asking "is this value's absence itself related to bankruptcy?"
**Why:** missingness isn't always random — if it correlates with the target, dropping or blindly imputing it throws away a signal. **Result:** 13 of 64 columns with missing values had missingness significantly related to the target (e.g. `profit_on_operating_activities_over_financial_expenses` was missing in 34.8% of bankrupt companies vs 5.0% of safe ones). These 13 get a `_was_missing` indicator feature later.

### 2. EDA (`03_eda.py`)
**What:** 17-part exploratory analysis — class-wise distributions, ratio-bound violations, skewness/kurtosis, zero-value checks, forecasting-period vs bankruptcy rate, correlation ranking, Mann-Whitney and Shapiro-Wilk tests, class-wise correlation structure, PCA/KMeans patterns, sign and inverse-ratio consistency checks.
**Why:** to understand the data before engineering features on it — e.g. it's what justifies adding an Altman Z-Score feature and pruning collinear ratios later, and it surfaces dataset quirks (like `"?"`-coded columns) that would otherwise silently corrupt a model.
**Result:** 24 figures + 10 summary tables in `reports/eda_tables/`.

### 3. Feature engineering (`04_feature_engineering.py`)
**What:** stratified 80/20 split → the 13 missing-indicator flags (fit on train only) → median imputation (train only) → 1st/99th percentile outlier capping → Altman Z-Score feature → iterative VIF pruning (drop most collinear column until all VIF ≤ 10) → 5 category-wise aggregate ratios (profitability, liquidity, leverage, efficiency, size) → one-hot encode forecasting period.
**Why:** raw financial ratios are extremely collinear (e.g. multiple "profit over X" ratios essentially repeat each other) and have wildly different scales/outliers — without this, tree models overweight redundant signal and non-tree models break on scale. Fitting everything on train only avoids leaking test information into preprocessing.
**Result:** 28 of 65 raw ratios dropped for multicollinearity (worst VIF was over 149 million). Every fitted value needed to repeat this transform on new data is saved to `preprocessing_artifacts.pkl`.

### 4. Model training (`05_model_Training.py`)
**What:** a validation split carved out of train (so the decision threshold is never touched by test data), 5-fold stratified CV for XGBoost and CatBoost (class-weighted for imbalance), then final metrics reported once on test.
**Why:** bankruptcy is rare in this data, so a naive 0.5 threshold and un-weighted training would just predict "safe" for almost everyone and still look accurate. Tuning the threshold on a separate validation split (not test) keeps the final test numbers honest.
**Result (test set, tuned threshold):** XGBoost — accuracy 0.969, precision 0.729, recall 0.502, F1 0.595, ROC-AUC 0.931. CatBoost — accuracy 0.969, precision 0.755, recall 0.467, F1 0.577, ROC-AUC 0.927.

### 5. Hyperparameter tuning (`06_tuning_threshold.py`)
**What:** Optuna search over XGBoost (50 trials, 5-fold CV, optimizing PR-AUC).
**Why:** to see if the default hyperparameters were leaving performance on the table.
**Result:** CV PR-AUC improved 0.643 → 0.666, but on the held-out test set the tuned model came in flat-to-slightly-worse than the baseline (F1 0.581 vs 0.595). So tuning helped in cross-validation but didn't clearly generalize better here — a useful thing to know before assuming "tuned" means "better."

### 6. SHAP explainability (`08_shap_explainability.py`)
**What:** `shap.TreeExplainer` on the tuned model — global importance, beeswarm summary, dependence plots for the top 3 features, waterfall plots for one correctly-flagged bankrupt and one correctly-flagged safe company, and a decision plot.
**Why:** a bankruptcy model that just outputs a number isn't very useful to a human decision-maker — SHAP shows *which ratios* pushed a specific company's risk up or down, which is what makes the dashboard's "why this prediction" panel possible.
**Result:** 6 figures + `shap_feature_importance.csv` ranking every feature by average impact and direction.

### 7. Serving (`app/inference.py` + `app/dashboard.py`)
**What:** `inference.py` replays the exact Step 3 transform on a new company's raw ratios using `preprocessing_artifacts.pkl`, scores it with the model, and returns the top SHAP factors. `dashboard.py` is a Streamlit UI on top of that (manual entry or CSV upload).
**Why:** so a new, unseen company can be scored the same way the model was trained, with no drift between training-time and prediction-time preprocessing.
**Note:** it currently loads `xgboost_baseline.pkl`, not `xgboost_tuned.pkl` — given Step 5's result, that's arguably the right call, but it isn't stated anywhere in the code.

### Not implemented
- `01_competitor_baseline.py` and `07_ensemble_stacking.py` are empty files, numbered into the pipeline but never written.

---

## Running it

```bash
pip install -r requirements.txt

python scripts/02_data_cleaning.py     # fix the hardcoded D:/ path first, or skip to the next line
python scripts/03_eda.py
python scripts/04_feature_engineering.py
python scripts/05_model_Training.py
python scripts/06_tuning_threshold.py
python scripts/08_shap_explainability.py

streamlit run app/dashboard.py
```

---

## Known issues / suggested next steps

- **`src/` is an empty package skeleton** — `pipeline.py`, `data/`, `features/`, `models/`, `explainability/` are all 0 bytes; all real logic lives in `scripts/` instead.
- **Empty scaffolding:** `requirements.txt` (filled in above), `configs/config.yaml`, `data/README.md`, `tests/test_pipeline.py` were all empty — no dependency pinning, no config-driven runs, no tests.
- **Hardcoded absolute path** in `02_data_cleaning.py` (`D:/Bankruptcy/...`) — every other script derives paths from `__file__` instead.
- **Recall is moderate (~0.50–0.59)** — for an early-warning system this means 40–50% of actually-bankrupt companies are still missed at the reported thresholds; worth optimizing for recall/F2 if catching more true positives matters more than false alarms.
- **~150MB of data/model files committed directly** — consider `.gitignore` + git-lfs/DVC for `data/` and `models/`.

---

## Tech stack

Python, pandas, scikit-learn, XGBoost, CatBoost, Optuna, SHAP, Streamlit, statsmodels, matplotlib.

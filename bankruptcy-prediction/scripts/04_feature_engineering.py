"""
04 - Feature Engineering
==========================
Order:
  1. Train/test split (stratified)
  2. Missing-indicator flags (fit on TRAIN)
  3. Median imputation (fit on TRAIN)
  4. Outlier capping/winsorizing (fit on TRAIN)
  5. Multicollinearity check (VIF) on RAW ratios ONLY - before derived features
  6. Altman Z-Score feature
  7. Category-wise aggregate features
  8. `forecasting period` one-hot encode
  9. Scaling (separate version, for non-tree models only)
  10. Combine train + test into one final CSV
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv')
OUT = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUT, exist_ok=True)

# ===========================================================
# 0. Load raw data
# ===========================================================
raw = pd.read_csv(INPUT_PATH, low_memory=False)
raw = raw.apply(pd.to_numeric, errors='coerce')

X = raw.drop(columns=['id', 'class'])
y = raw['class']

# ===========================================================
# 1. Train/test split (stratified)
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

ratio_cols = [c for c in X.columns if c != 'forecasting period']

# ===========================================================
# 2. Missing-indicator flags (fit on TRAIN only)
# ===========================================================
indicator_cols = []
for col in ratio_cols:
    if X_train[col].isna().sum() == 0:
        continue
    is_missing = X_train[col].isna().astype(int)
    ct = pd.crosstab(is_missing, y_train)
    if ct.shape[0] < 2:
        continue
    _, p, _, _ = stats.chi2_contingency(ct)
    if p < 0.05:
        indicator_cols.append(col)

for col in indicator_cols:
    X_train[f'{col}_was_missing'] = X_train[col].isna().astype(int)
    X_test[f'{col}_was_missing'] = X_test[col].isna().astype(int)

print(f"Missing-indicator flags added: {len(indicator_cols)}")

# ===========================================================
# 3. Median imputation (fit on TRAIN, applied to both)
# ===========================================================
medians = X_train[ratio_cols].median()
X_train[ratio_cols] = X_train[ratio_cols].fillna(medians)
X_test[ratio_cols] = X_test[ratio_cols].fillna(medians)
print("Median imputation done (train medians applied to test).")

# ===========================================================
# 4. Outlier capping / winsorizing (fit on TRAIN)
# ===========================================================
lower = X_train[ratio_cols].quantile(0.01)
upper = X_train[ratio_cols].quantile(0.99)

X_train_capped = X_train.copy()
X_test_capped = X_test.copy()
X_train_capped[ratio_cols] = X_train[ratio_cols].clip(lower, upper, axis=1)
X_test_capped[ratio_cols] = X_test[ratio_cols].clip(lower, upper, axis=1)
print("Outlier capping done (1st-99th percentile).")

# ===========================================================
# 5. Altman Z-Score feature (calculated BEFORE VIF-based
#    dropping, since it needs specific raw ratio columns that
#    the VIF step might otherwise remove as "redundant")
# ===========================================================
def add_altman_z(df_):
    df_ = df_.copy()
    df_['altman_z'] = (
        1.2 * df_['working_capital_over_total_assets'] +
        1.4 * df_['retained_earnings_over_total_assets'] +
        3.3 * df_['EBIT_over_total_assets'] +
        0.6 * df_['book_value_of_equity_over_total_liabilities'] +
        1.0 * df_['sales_over_total_assets']
    )
    return df_

X_train = add_altman_z(X_train)
X_test = add_altman_z(X_test)
X_train_capped = add_altman_z(X_train_capped)
X_test_capped = add_altman_z(X_test_capped)
print("Altman Z-Score feature added.")

# ===========================================================
# 6. Multicollinearity check (VIF) - FIXED VERSION
# ===========================================================
def drop_high_vif(df_, threshold=10, max_iter=30):
    """Iteratively drop the single highest-VIF column until every
    remaining column has VIF <= threshold (or max_iter is reached)."""
    cols = df_.columns.tolist()
    dropped = []
    for _ in range(max_iter):
        vif_vals = pd.Series(
            [variance_inflation_factor(df_[cols].values, i) for i in range(len(cols))],
            index=cols
        )
        max_vif = vif_vals.max()
        if max_vif > threshold and np.isfinite(max_vif):
            worst_col = vif_vals.idxmax()
            dropped.append((worst_col, max_vif))
            cols.remove(worst_col)
        else:
            break
    return cols, dropped

kept_ratio_cols, dropped_log = drop_high_vif(X_train_capped[ratio_cols], threshold=10)
print(f"VIF check: dropped {len(dropped_log)} / {len(ratio_cols)} high-multicollinearity features.")
for col, v in dropped_log:
    print(f"   dropped: {col}  (VIF={v:.1f})")

# Actually DROP the flagged columns (altman_z is untouched - it's not in ratio_cols).
dropped_cols = [col for col, v in dropped_log]

X_train = X_train.drop(columns=dropped_cols)
X_test = X_test.drop(columns=[c for c in dropped_cols if c in X_test.columns])
X_train_capped = X_train_capped.drop(columns=dropped_cols)
X_test_capped = X_test_capped.drop(columns=[c for c in dropped_cols if c in X_test_capped.columns])

ratio_cols = kept_ratio_cols

# ===========================================================
# 7. Category-wise aggregate features
# ===========================================================
def categorize(col):
    c = col.lower()
    if any(k in c for k in ['profit', 'ebit', 'ebitda', 'earnings']):
        return 'profitability'
    if any(k in c for k in ['current_assets', 'cash', 'working_capital', 'liquidit']):
        return 'liquidity'
    if any(k in c for k in ['liabilit', 'equity', 'capital', 'debt']):
        return 'leverage'
    if any(k in c for k in ['sales', 'inventory', 'receivable', 'turnover', 'rotation', 'cost']):
        return 'efficiency'
    return 'size'

cat_map = {c: categorize(c) for c in ratio_cols}

def add_category_aggregates(df_):
    df_ = df_.copy()
    for cat in sorted(set(cat_map.values())):
        cols_in_cat = [c for c, v in cat_map.items() if v == cat]
        df_[f'agg_{cat}_mean'] = df_[cols_in_cat].mean(axis=1)
    return df_

X_train = add_category_aggregates(X_train)
X_test = add_category_aggregates(X_test)
X_train_capped = add_category_aggregates(X_train_capped)
X_test_capped = add_category_aggregates(X_test_capped)
print("Category-wise aggregate features added.")

# ===========================================================
# 8. `forecasting period` one-hot encode
# ===========================================================
def one_hot_period(df_):
    return pd.get_dummies(df_, columns=['forecasting period'], prefix='period', dtype=int)

X_train = one_hot_period(X_train)
X_test = one_hot_period(X_test)
X_train_capped = one_hot_period(X_train_capped)
X_test_capped = one_hot_period(X_test_capped)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_test_capped = X_test_capped.reindex(columns=X_train_capped.columns, fill_value=0)
print("`forecasting period` one-hot encoded.")

# ===========================================================
# 9. Scaling (separate scaled version, non-tree models only)
# ===========================================================
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_capped), columns=X_train_capped.columns, index=X_train_capped.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_capped), columns=X_test_capped.columns, index=X_test_capped.index
)
print("Scaling done (fit on train only) - use for Logistic Regression/KMeans.")

# ===========================================================
# Save everything
# ===========================================================
X_train.assign(**{'class': y_train.values}).to_csv(f'{OUT}/train_features.csv', index=False)
X_test.assign(**{'class': y_test.values}).to_csv(f'{OUT}/test_features.csv', index=False)

X_train_capped.assign(**{'class': y_train.values}).to_csv(f'{OUT}/train_features_capped.csv', index=False)
X_test_capped.assign(**{'class': y_test.values}).to_csv(f'{OUT}/test_features_capped.csv', index=False)

X_train_scaled.assign(**{'class': y_train.values}).to_csv(f'{OUT}/train_features_scaled.csv', index=False)
X_test_scaled.assign(**{'class': y_test.values}).to_csv(f'{OUT}/test_features_scaled.csv', index=False)

pd.DataFrame(dropped_log, columns=['feature', 'vif']).to_csv(f'{OUT}/vif_dropped_features.csv', index=False)

print(f"\nAll feature-engineered files saved to: {OUT}")
print(f"Final feature count (tree models): {X_train.shape[1]}")

# ===========================================================
# 10. Combine train + test into ONE final CSV (capped version -
#     ready for tree models: XGBoost / LightGBM / CatBoost)
# ===========================================================
train_final = X_train_capped.copy()
train_final['class'] = y_train.values
train_final['split'] = 'train'

test_final = X_test_capped.copy()
test_final['class'] = y_test.values
test_final['split'] = 'test'

final_df = pd.concat([train_final, test_final], axis=0, ignore_index=True)
final_df.to_csv(f'{OUT}/features_final.csv', index=False)

print(f"\nSaved combined file: {OUT}/features_final.csv")
print(f"Final shape: {final_df.shape}  (train={len(train_final)}, test={len(test_final)})")



# ===========================================================
# 11. Save preprocessing artifacts (needed for inference on NEW data)
# ===========================================================
# Why: everything above (medians, capping bounds, which columns
# survived the VIF check, the category mapping, the final column
# order) only exists in memory right now. Without saving it, there
# is no way for the app/product later to turn a brand-new company's
# raw ratios into the exact 57-feature vector this model expects.

import joblib

artifacts = {
    'ratio_cols_original': [c for c in X.columns if c != 'forecasting period'],
    'indicator_cols': indicator_cols,          # which columns get a _was_missing flag
    'medians': medians.to_dict(),              # imputation values (from train only)
    'lower_bounds': lower.to_dict(),           # outlier-capping lower bound (1st pct, train only)
    'upper_bounds': upper.to_dict(),           # outlier-capping upper bound (99th pct, train only)
    'kept_ratio_cols': ratio_cols,             # ratio columns that survived VIF (final 57-set uses these)
    'cat_map': cat_map,                        # feature -> category, for the aggregate features
    'period_values': sorted(X['forecasting period'].dropna().unique().tolist()),  # for one-hot encoding
    'final_feature_columns': X_train_capped.columns.tolist(),  # exact column order the model expects
    'scale_pos_weight': float((y_train == 0).sum() / (y_train == 1).sum()),
}

joblib.dump(artifacts, f'{OUT}/preprocessing_artifacts.pkl')
print(f"\nSaved preprocessing artifacts to: {OUT}/preprocessing_artifacts.pkl")
print("This file lets app/inference.py transform a NEW company's raw ratios "
      "into the exact feature vector the trained model expects.")



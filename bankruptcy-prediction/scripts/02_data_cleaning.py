"""
02 - Data Cleaning (diagnosis only)

Scope: load data, basic structural checks, numeric conversion,
missing value detection, and missingness randomness check.

Nothing is filled/imputed yet - this script only diagnoses the data.
"""

import pandas as pd
from scipy import stats

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)


# ---------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------
df = pd.read_csv('D:/Bankruptcy/bankruptcy-prediction/data/raw/train.csv', low_memory=False)
print("Shape:", df.shape)


# ---------------------------------------------------------
# 2. Basic structural checks - shape, info, duplicates
# ---------------------------------------------------------
print("\n--- Column info ---")
df.info()

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate ids:", df['id'].duplicated().sum())


# ---------------------------------------------------------
# 3. Numeric conversion
# ---------------------------------------------------------
# Raw columns are stored as text. Convert to numeric -
# anything that fails to convert becomes NaN.
num_df = df.apply(pd.to_numeric, errors='coerce')

print("\n--- Describe (after numeric conversion) ---")
print(num_df.describe().T)


# ---------------------------------------------------------
# 4. Missing value detection & quantification
# ---------------------------------------------------------
missing_count = num_df.isna().sum().sort_values(ascending=False)
missing_pct = (missing_count / len(num_df) * 100).round(2)

missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_pct': missing_pct
})
missing_summary = missing_summary[missing_summary['missing_count'] > 0]

print(f"\nTotal NaN cells: {num_df.isna().sum().sum()}")
print(f"Rows with at least one NaN: {num_df.isna().any(axis=1).sum()} / {len(num_df)}")
print(f"Columns with missing values: {len(missing_summary)} / {num_df.shape[1]}")
print("\n--- Missing value summary ---")
print(missing_summary)


# ---------------------------------------------------------
# 5. Missingness randomness check (chi-square vs target)
# ---------------------------------------------------------
# For every column with missing values, test whether
# "is this value missing" (0/1) is statistically related
# to the target `class`.
#
# p >= 0.05 -> missingness looks random w.r.t. bankruptcy status
# p < 0.05  -> missingness is related to bankruptcy status
#              (it's a signal, not just noise)
results = []
for col in missing_summary.index:
    is_missing = num_df[col].isna().astype(int)
    ct = pd.crosstab(is_missing, num_df['class'])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    miss_rate_bankrupt = num_df.loc[num_df['class'] == 1, col].isna().mean() * 100
    miss_rate_safe = num_df.loc[num_df['class'] == 0, col].isna().mean() * 100
    results.append({
        'column': col,
        'missing_pct_overall': missing_summary.loc[col, 'missing_pct'],
        'missing_pct_bankrupt': round(miss_rate_bankrupt, 2),
        'missing_pct_safe': round(miss_rate_safe, 2),
        'chi2_pvalue': p,
        'related_to_target': p < 0.05
    })

missingness_test = pd.DataFrame(results).sort_values('chi2_pvalue')
print("\n--- Missingness vs target (chi-square test) ---")
print(missingness_test)

n_related = missingness_test['related_to_target'].sum()
n_total = len(missingness_test)
print(f"\n{n_related} / {n_total} columns have missingness significantly "
      f"related to the target (p < 0.05).")

cols_needing_indicator = missingness_test.loc[
    missingness_test['related_to_target'], 'column'
].tolist()
print("\nColumns needing a missing-indicator feature:")
print(cols_needing_indicator)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n=== Summary ===")
print(f"- Duplicate rows: {df.duplicated().sum()}")
print(f"- {len(missing_summary)} / {num_df.shape[1]} columns have missing values.")
print(f"- {n_related} / {n_total} of those columns have missingness related "
      f"to the target (not random).")
print("- These are flagged for missing-indicator treatment in the next step "
      "(filling/imputation) - not done in this script.")


# ---------------------------------------------------------
# 6. Missing-indicator features (only for the 13 non-random columns)
# ---------------------------------------------------------
# These are the columns whose missingness was significantly
# related to the target (p < 0.05) - identified in Step 5.
indicator_cols = cols_needing_indicator

for col in indicator_cols:
    num_df[f'{col}_was_missing'] = num_df[col].isna().astype(int)

print(f"\nAdded {len(indicator_cols)} indicator columns.")


# ---------------------------------------------------------
# 7. Fill missing values (median, applied to all 64 columns)
# ---------------------------------------------------------
feature_cols = [c for c in num_df.columns if c not in ('id', 'class') and '_was_missing' not in c]

for col in feature_cols:
    if num_df[col].isna().sum() > 0:
        median_val = num_df[col].median()
        num_df[col] = num_df[col].fillna(median_val)

print("Missing values filled with column median.")


# ---------------------------------------------------------
# 8. Verify no NaN remains
# ---------------------------------------------------------
remaining_nan = num_df.isna().sum().sum()
print(f"\nRemaining NaN cells after fill: {remaining_nan}")
print("Final shape:", num_df.shape, "(original 67 + 13 indicator columns = 80)")


# ---------------------------------------------------------
# 9. Save cleaned dataset
# ---------------------------------------------------------
import os
os.makedirs('../data/processed', exist_ok=True)
num_df.to_csv('../data/processed/train_clean.csv', index=False)
print("\nSaved: data/processed/train_clean.csv")


# ---------------------------------------------------------
# 10. Outlier detection (IQR method) - detect only, no removal
# ---------------------------------------------------------
# For each feature, use the IQR rule:
#   lower_bound = Q1 - 1.5 * IQR
#   upper_bound = Q3 + 1.5 * IQR
# Any value outside this range is flagged as a statistical outlier.

outlier_report = []

for col in feature_cols:
    series = num_df[col]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    is_outlier = (series < lower) | (series > upper)
    outlier_count = is_outlier.sum()
    outlier_pct = round(outlier_count / len(series) * 100, 2)

    outlier_report.append({
        'column': col,
        'outlier_count': outlier_count,
        'outlier_pct': outlier_pct
    })

outlier_df = pd.DataFrame(outlier_report).sort_values('outlier_pct', ascending=False)

print("\n--- Outlier detection (IQR method) ---")
print(outlier_df)
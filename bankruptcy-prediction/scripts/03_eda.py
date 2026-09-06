"""
FULL EDA - Bankruptcy Prediction Dataset
==========================================
Combines:
  PART A - Data Cleaning (diagnosis + median-fill + missing indicators)
  PART B - Original 9 EDA steps
           1. Class-wise comparison (boxplots)
           2. Feature distributions (histograms)
           3. Theoretically-bounded ratio violations
           4. Skewness / Kurtosis quantification
           5. Zero-value analysis
           6. `forecasting period` vs bankruptcy rate
           7. Target correlation ranking
           8. Statistical significance test per feature (Mann-Whitney U)
           9. Normality test (Shapiro-Wilk)
  PART C - 8 Advanced/Specific EDA steps
           10. Class-wise correlation structure comparison
           11. Multivariate pattern (scatter)
           12. Ratio category-wise grouping
           13. Sign-based observation (negative values)
           14. Inverse-pair consistency check
           15. "?" values - specific column check (on RAW data)
           16. logarithm_of_total_assets - special observation
           17. id column - sequential pattern check

All figures: grid OFF, distinct colors per series/bar/slope, clear legends & annotations.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


warnings.filterwarnings('ignore')

# ---------------- Global style ----------------
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

COLOR_SAFE = '#2E86AB'      # blue
COLOR_BANKRUPT = '#E63946'  # red
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)   # goes up from /scripts to project root

INPUT_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv')
OUT = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(OUT, exist_ok=True)

print("Saving figures to:", OUT)   # add this once to confirm the correct path

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 200)


# ===========================================================
# PART A - DATA CLEANING (diagnosis + indicators + median fill)
# ===========================================================
raw = pd.read_csv(INPUT_PATH, low_memory=False)
print("Shape:", raw.shape)

print("\nDuplicate rows:", raw.duplicated().sum())
print("Duplicate ids:", raw['id'].duplicated().sum())

num_df = raw.apply(pd.to_numeric, errors='coerce')

missing_count = num_df.isna().sum().sort_values(ascending=False)
missing_pct = (missing_count / len(num_df) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing_count, 'missing_pct': missing_pct})
missing_summary = missing_summary[missing_summary['missing_count'] > 0]

print(f"\nTotal NaN cells: {num_df.isna().sum().sum()}")
print(f"Rows with at least one NaN: {num_df.isna().any(axis=1).sum()} / {len(num_df)}")
print(f"Columns with missing values: {len(missing_summary)} / {num_df.shape[1]}")

# Missingness vs target (chi-square)
results = []
for col in missing_summary.index:
    if col == 'class':
        continue
    is_missing = num_df[col].isna().astype(int)
    ct = pd.crosstab(is_missing, num_df['class'])
    if ct.shape[0] < 2:
        continue
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    miss_rate_bankrupt = num_df.loc[num_df['class'] == 1, col].isna().mean() * 100
    miss_rate_safe = num_df.loc[num_df['class'] == 0, col].isna().mean() * 100
    results.append({'column': col, 'missing_pct_overall': missing_summary.loc[col, 'missing_pct'],
                     'missing_pct_bankrupt': round(miss_rate_bankrupt, 2),
                     'missing_pct_safe': round(miss_rate_safe, 2),
                     'chi2_pvalue': p, 'related_to_target': p < 0.05})

missingness_test = pd.DataFrame(results)
if len(missingness_test) > 0:
    missingness_test = missingness_test.sort_values('chi2_pvalue')
    n_related = missingness_test['related_to_target'].sum()
    n_total = len(missingness_test)
    print(f"\n{n_related} / {n_total} columns have missingness significantly related to the target (p < 0.05).")
    cols_needing_indicator = missingness_test.loc[missingness_test['related_to_target'], 'column'].tolist()
else:
    print("\nNo columns had missing values, so no missingness-vs-target test was needed.")
    cols_needing_indicator = []

for col in cols_needing_indicator:
    num_df[f'{col}_was_missing'] = num_df[col].isna().astype(int)
print(f"Added {len(cols_needing_indicator)} missing-indicator columns.")

feature_cols = [c for c in num_df.columns if c not in ('id', 'forecasting period', 'class') and '_was_missing' not in c]

for col in feature_cols:
    if num_df[col].isna().sum() > 0:
        num_df[col] = num_df[col].fillna(num_df[col].median())

remaining_nan = num_df.isna().sum().sum()
print(f"Remaining NaN cells after median fill: {remaining_nan}")
print("Cleaned shape:", num_df.shape)

df = num_df  # cleaned dataframe used for all EDA below
safe_mask = df['class'] == 0
bankrupt_mask = df['class'] == 1

corr_with_target = df[feature_cols].apply(lambda x: x.corr(df['class'])).abs().sort_values(ascending=False)

mw_pvals = {}
for col in feature_cols:
    s = df.loc[safe_mask, col].dropna()
    b = df.loc[bankrupt_mask, col].dropna()
    if len(s) > 5 and len(b) > 5:
        try:
            _, p = stats.mannwhitneyu(s, b, alternative='two-sided')
            mw_pvals[col] = p
        except Exception:
            pass
mw_rank = pd.Series(mw_pvals).sort_values()

print("\n=== PART A complete: data cleaned in memory, ready for EDA ===\n")


# ===========================================================
# PART B - ORIGINAL 9 EDA STEPS
# ===========================================================

# -----------------------------------------------------------
# B1. Class-wise comparison (boxplots)
# -----------------------------------------------------------
top12 = corr_with_target.head(12).index.tolist()

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
fig.suptitle('Feature Distributions by Bankruptcy Status (Top 12 by Correlation with Target)',
             fontsize=15, fontweight='bold', y=1.03)

for ax, col in zip(axes.flatten(), top12):
    data_safe = df.loc[df['class'] == 0, col]
    data_bankrupt = df.loc[df['class'] == 1, col]
    lo, hi = df[col].quantile([0.01, 0.99])
    data_safe_c = data_safe.clip(lo, hi)
    data_bankrupt_c = data_bankrupt.clip(lo, hi)

    bp = ax.boxplot([data_safe_c, data_bankrupt_c], tick_labels=['Safe', 'Bankrupt'],
                     patch_artist=True, widths=0.5, showfliers=False)
    bp['boxes'][0].set_facecolor(COLOR_SAFE)
    bp['boxes'][1].set_facecolor(COLOR_BANKRUPT)
    for box in bp['boxes']:
        box.set_alpha(0.85)
        box.set_edgecolor('black')
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)

    short_name = col if len(col) <= 24 else col[:21] + '...'
    ax.set_title(short_name, fontsize=9.5, wrap=True)
    ax.tick_params(axis='both', labelsize=8.5)

legend_handles = [
    mpatches.Patch(facecolor=COLOR_SAFE, alpha=0.85, edgecolor='black', label='Safe company'),
    mpatches.Patch(facecolor=COLOR_BANKRUPT, alpha=0.85, edgecolor='black', label='Bankrupt company')
]
fig.legend(handles=legend_handles, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.0),
           fontsize=11.5, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f'{OUT}/B1_classwise_boxplots.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B1_classwise_boxplots.png")


# -----------------------------------------------------------
# B2. Feature distributions (histograms)
# -----------------------------------------------------------
top9 = corr_with_target.head(9).index.tolist()

fig, axes = plt.subplots(3, 3, figsize=(17, 13))
fig.suptitle('Feature Distributions - Safe vs Bankrupt (Top 9 by Correlation)',
             fontsize=15, fontweight='bold', y=1.02)

for ax, col in zip(axes.flatten(), top9):
    lo, hi = df[col].quantile([0.02, 0.98])
    safe_vals = df.loc[df['class'] == 0, col].clip(lo, hi)
    bankrupt_vals = df.loc[df['class'] == 1, col].clip(lo, hi)

    ax.hist(safe_vals, bins=40, alpha=0.65, color=COLOR_SAFE, label='Safe', density=True,
            edgecolor='white', linewidth=0.3)
    ax.hist(bankrupt_vals, bins=40, alpha=0.65, color=COLOR_BANKRUPT, label='Bankrupt', density=True,
            edgecolor='white', linewidth=0.3)
    ax.axvline(safe_vals.median(), color=COLOR_SAFE, linestyle='--', linewidth=1.6)
    ax.axvline(bankrupt_vals.median(), color=COLOR_BANKRUPT, linestyle='--', linewidth=1.6)

    short_name = col if len(col) <= 26 else col[:23] + '...'
    ax.set_title(short_name, fontsize=10.5)
    ax.set_ylabel('Density', fontsize=8.5)
    leg = ax.legend(fontsize=8, loc='upper right', frameon=True)
    leg.get_frame().set_edgecolor('black')
    ax.tick_params(labelsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f'{OUT}/B2_feature_distributions.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B2_feature_distributions.png")


# -----------------------------------------------------------
# B3. Theoretically-bounded ratio violations
# -----------------------------------------------------------
bounded_candidates = [c for c in feature_cols if 'over_total_assets' in c or 'over_total_liabilities' in c]
violations = {}
for col in bounded_candidates:
    below0 = (df[col] < 0).sum()
    above1 = (df[col] > 1).sum()
    violations[col] = {'below_0': below0, 'above_1': above1,
                        'pct_violating': round((below0 + above1) / len(df) * 100, 2)}

violations_df = pd.DataFrame(violations).T.sort_values('pct_violating', ascending=False)
top_v = violations_df.head(15)

fig, ax = plt.subplots(figsize=(12, 8))
labels_v = [c if len(c) <= 32 else c[:29] + '...' for c in top_v.index]
y_pos = np.arange(len(top_v))
b1 = ax.barh(y_pos, top_v['below_0'], color='#457B9D', edgecolor='black', linewidth=0.6, label='Below 0 (violates lower bound)')
b2 = ax.barh(y_pos, top_v['above_1'], left=top_v['below_0'], color='#E76F51', edgecolor='black',
             linewidth=0.6, label='Above 1 (violates upper bound)')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_v, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of violating rows', fontsize=10.5)
ax.set_title('Theoretically-Bounded Ratios [0,1] — How Often Are They Violated?', fontsize=13, pad=12)
for i, (b0, a1, pct) in enumerate(zip(top_v['below_0'], top_v['above_1'], top_v['pct_violating'])):
    ax.text(b0 + a1 + max(top_v['below_0']+top_v['above_1'])*0.01, i, f'{pct:.1f}%',
            va='center', fontsize=8.5, fontweight='bold')
legend = ax.legend(loc='lower right', fontsize=9.5, frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/B3_bounded_ratio_violations.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B3_bounded_ratio_violations.png")


# -----------------------------------------------------------
# B4. Skewness / Kurtosis quantification
# -----------------------------------------------------------
skew_kurt = pd.DataFrame({
    'skewness': df[feature_cols].skew(),
    'kurtosis': df[feature_cols].kurtosis()
}).sort_values('skewness', ascending=False)

top_pos = skew_kurt.head(10)
top_neg = skew_kurt.tail(10)
combo = pd.concat([top_pos, top_neg])
labels_sk = [c if len(c) <= 30 else c[:27] + '...' for c in combo.index]
colors_sk = ['#E76F51'] * len(top_pos) + ['#457B9D'] * len(top_neg)

fig, ax = plt.subplots(figsize=(12, 9))
y_pos = np.arange(len(combo))
bars = ax.barh(y_pos, combo['skewness'], color=colors_sk, edgecolor='black', linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_sk, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Skewness', fontsize=10.5)
ax.set_title('Most Positively & Negatively Skewed Features', fontsize=13, pad=12)
legend_patches = [mpatches.Patch(color='#E76F51', label='Top 10 most positively skewed'),
                  mpatches.Patch(color='#457B9D', label='Top 10 most negatively skewed')]
legend = ax.legend(handles=legend_patches, loc='lower right', fontsize=9.5, frameon=True)
legend.get_frame().set_edgecolor('black')
for i, val in enumerate(combo['skewness']):
    ax.text(val + (2 if val >= 0 else -2), i, f'{val:.1f}', va='center',
            ha='left' if val >= 0 else 'right', fontsize=7.5)
plt.tight_layout()
plt.savefig(f'{OUT}/B4_skewness_kurtosis.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B4_skewness_kurtosis.png")


# -----------------------------------------------------------
# B5. Zero-value analysis
# -----------------------------------------------------------
zero_pct = (df[feature_cols] == 0).mean().sort_values(ascending=False) * 100
zero_pct = zero_pct[zero_pct > 0].head(15)

fig, ax = plt.subplots(figsize=(11, 6.5))
if len(zero_pct) == 0:
    ax.text(0.5, 0.5, 'No feature contains any exact-zero values\n(expected for continuous financial ratios).',
            ha='center', va='center', fontsize=14, fontweight='bold', color='#2A9D8F',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F6F3', edgecolor='#2A9D8F'))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Zero-Value Analysis', fontsize=13, pad=12)
else:
    labels_z = [c if len(c) <= 30 else c[:27] + '...' for c in zero_pct.index]
    colors_z = plt.cm.plasma(np.linspace(0.15, 0.85, len(zero_pct)))
    bars = ax.barh(range(len(zero_pct)), zero_pct.values[::-1], color=colors_z[::-1],
                   edgecolor='black', linewidth=0.6)
    ax.set_yticks(range(len(zero_pct)))
    ax.set_yticklabels(labels_z[::-1], fontsize=9)
    ax.set_xlabel('% of exact-zero values', fontsize=10.5)
    ax.set_title('Top 15 Columns by % of Exact-Zero Values', fontsize=13, pad=12)
    for i, v in enumerate(zero_pct.values[::-1]):
        ax.text(v + 0.05, i, f'{v:.2f}%', va='center', fontsize=8.5, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/B5_zero_value_analysis.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B5_zero_value_analysis.png")


# -----------------------------------------------------------
# B6. `forecasting period` vs bankruptcy rate
# -----------------------------------------------------------
period_rates = df.groupby('forecasting period')['class'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 6))
period_palette = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
bars = ax.bar(period_rates.index.astype(str), period_rates.values,
              color=period_palette[:len(period_rates)], alpha=0.95, width=0.55, edgecolor='black', linewidth=1)
ax.set_xlabel('Forecasting period', fontsize=10.5)
ax.set_ylabel('Bankruptcy rate (%)', fontsize=10.5)
ax.set_title('Bankruptcy Rate by Forecasting Period', fontsize=13, pad=12)
for bar, val in zip(bars, period_rates.values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.08, f'{val:.2f}%',
            ha='center', fontsize=10, fontweight='bold')
legend_handles = [mpatches.Patch(facecolor=c, edgecolor='black', label=f'Period {int(p)}')
                  for p, c in zip(period_rates.index, period_palette[:len(period_rates)])]
legend = ax.legend(handles=legend_handles, loc='upper left', fontsize=9, frameon=True)
legend.get_frame().set_edgecolor('black')
ax.annotate('Bankruptcy risk rises as the\nforecast horizon lengthens',
            xy=(len(period_rates)-1, period_rates.values[-1]), xytext=(0.05, 0.85),
            textcoords='axes fraction', fontsize=10, fontweight='bold', color='#E76F51',
            arrowprops=dict(arrowstyle='->', color='#E76F51', lw=1.4),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#E76F51'))
plt.tight_layout()
plt.savefig(f'{OUT}/B6_period_vs_bankruptcy_rate.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B6_period_vs_bankruptcy_rate.png")


# -----------------------------------------------------------
# B7. Target correlation ranking
# -----------------------------------------------------------
top20 = corr_with_target.head(20)

fig, ax = plt.subplots(figsize=(11, 9.5))
palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(top20)))
bars = ax.barh(range(len(top20)), top20.values[::-1], color=palette[::-1], edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels([c if len(c) <= 40 else c[:37] + '...' for c in top20.index[::-1]], fontsize=8.5)
ax.set_xlabel('Absolute correlation with bankruptcy (class)', fontsize=10.5)
ax.set_title('Top 20 Features by Correlation with Target', fontsize=13, pad=12)
for i, v in enumerate(top20.values[::-1]):
    ax.text(v + 0.0008, i, f'{v:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/B7_target_correlation_ranking.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B7_target_correlation_ranking.png")


# -----------------------------------------------------------
# B8. Statistical significance test per feature (Mann-Whitney U)
# -----------------------------------------------------------
mw_df = mw_rank.reset_index()
mw_df.columns = ['column', 'p_value']
mw_df['significant'] = mw_df['p_value'] < 0.05
n_sig = mw_df['significant'].sum()
print(f"\n--- Mann-Whitney U test ---")
print(f"{n_sig} / {len(mw_df)} features show a statistically significant difference "
      f"between safe and bankrupt companies (p < 0.05).")

top10_sig = mw_df.head(10)
bottom10_sig = mw_df.tail(10)
combo_mw = pd.concat([top10_sig, bottom10_sig])
labels_mw = [c if len(c) <= 30 else c[:27] + '...' for c in combo_mw['column']]
colors_mw = ['#2A9D8F' if s else '#E76F51' for s in combo_mw['significant']]

fig, ax = plt.subplots(figsize=(12, 9))
y_pos = np.arange(len(combo_mw))
neg_log_p = -np.log10(combo_mw['p_value'].clip(lower=1e-300))
bars = ax.barh(y_pos, neg_log_p, color=colors_mw, edgecolor='black', linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels_mw, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.4)
ax.set_xlabel('-log10(p-value)  [higher = more significant]', fontsize=10.5)
ax.set_title('Mann-Whitney U Test: Most vs Least Discriminative Features', fontsize=13, pad=12)
legend_patches = [mpatches.Patch(color='#2A9D8F', label='Significant (p<0.05)'),
                  mpatches.Patch(color='#E76F51', label='Not significant'),
                  plt.Line2D([0], [0], color='black', linestyle='--', label='p = 0.05 threshold')]
legend = ax.legend(handles=legend_patches, loc='lower right', fontsize=9, frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/B8_mannwhitney_significance.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B8_mannwhitney_significance.png")


# -----------------------------------------------------------
# B9. Normality test (Shapiro-Wilk)
# -----------------------------------------------------------
np.random.seed(42)
sample_idx = np.random.choice(df.index, size=500, replace=False)

normality_results = []
for col in feature_cols:
    stat, p = stats.shapiro(df.loc[sample_idx, col])
    normality_results.append({'column': col, 'shapiro_p': p, 'is_normal': p > 0.05})

norm_df = pd.DataFrame(normality_results).sort_values('shapiro_p', ascending=False)
n_normal = norm_df['is_normal'].sum()
print(f"\n--- Shapiro-Wilk normality test (n=500 sample) ---")
print(f"{n_normal} / {len(norm_df)} features look normally distributed (p > 0.05).")

fig, ax = plt.subplots(figsize=(8, 6))
counts = [n_normal, len(norm_df) - n_normal]
labels_n = [f'Normal\n(p>0.05)\nn={n_normal}', f'Not Normal\n(p<0.05)\nn={len(norm_df)-n_normal}']
colors_n = ['#2A9D8F', '#E76F51']
wedges, texts, autotexts = ax.pie(counts, labels=labels_n, colors=colors_n, autopct='%1.1f%%',
                                    startangle=90, wedgeprops=dict(edgecolor='black', linewidth=1.2),
                                    textprops=dict(fontsize=10.5, fontweight='bold'))
ax.set_title(f'Shapiro-Wilk Normality Test\n(n=500 sample, {len(norm_df)} features tested)', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f'{OUT}/B9_shapiro_normality.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: B9_shapiro_normality.png")

print("\n=== PART B complete (9 original EDA steps) ===\n")


# ===========================================================
# PART C - 8 ADVANCED / DATASET-SPECIFIC EDA STEPS
# ===========================================================

# -----------------------------------------------------------
# C1. Class-wise correlation structure comparison
# -----------------------------------------------------------
top15 = corr_with_target.head(15).index.tolist()
short_labels = [c if len(c) <= 22 else c[:19] + '...' for c in top15]

corr_safe = df.loc[df['class'] == 0, top15].corr()
corr_bankrupt = df.loc[df['class'] == 1, top15].corr()

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Class-wise Correlation Structure: Do Bankrupt Companies Behave Differently?',
             fontsize=16, fontweight='bold', y=1.02)

for ax, corr_mat, title, cmap in [
    (axes[0], corr_safe, 'SAFE Companies — Feature Correlation', 'Blues'),
    (axes[1], corr_bankrupt, 'BANKRUPT Companies — Feature Correlation', 'Reds')
]:
    im = ax.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_title(title, fontsize=13, pad=10)
    for i in range(len(short_labels)):
        for j in range(len(short_labels)):
            val = corr_mat.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation coefficient', fontsize=9)

diff = (corr_bankrupt - corr_safe).abs().copy()
diff_vals = diff.values.copy()
np.fill_diagonal(diff_vals, 0)
diff = pd.DataFrame(diff_vals, index=diff.index, columns=diff.columns)
max_diff_val = diff.values.max()
max_idx = np.unravel_index(diff.values.argmax(), diff.values.shape)
fig.text(0.5, -0.03,
          f"Insight: The largest structural shift is between "
          f"'{short_labels[max_idx[0]]}' and '{short_labels[max_idx[1]]}' "
          f"(|Delta correlation| = {max_diff_val:.2f}) — the relationship between these two ratios "
          f"changes noticeably once a company becomes distressed.",
          ha='center', fontsize=10.5, style='italic',
          bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF3CD', edgecolor='#E9C46A'))

plt.tight_layout()
plt.savefig(f'{OUT}/C1_classwise_correlation_structure.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C1_classwise_correlation_structure.png")


# -----------------------------------------------------------
# C2. Multivariate pattern (2-3 features together, scatter)
# -----------------------------------------------------------
top3 = mw_rank.head(3).index.tolist()
fx, fy, fz = top3

lo_x, hi_x = df[fx].quantile([0.02, 0.98])
lo_y, hi_y = df[fy].quantile([0.02, 0.98])

plot_df = df[[fx, fy, fz, 'class']].dropna().copy()
plot_df[fx] = plot_df[fx].clip(lo_x, hi_x)
plot_df[fy] = plot_df[fy].clip(lo_y, hi_y)

safe_pts = plot_df[plot_df['class'] == 0]
bankrupt_pts = plot_df[plot_df['class'] == 1]

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.scatter(safe_pts[fx], safe_pts[fy], c=COLOR_SAFE, s=18, alpha=0.35,
           edgecolors='none', label=f'Safe (n={len(safe_pts):,})', zorder=2)
ax.scatter(bankrupt_pts[fx], bankrupt_pts[fy], c=COLOR_BANKRUPT, s=32, alpha=0.75,
           edgecolors='black', linewidths=0.3, label=f'Bankrupt (n={len(bankrupt_pts):,})', zorder=3)

ax.set_xlabel(fx.replace('_', ' ').title(), fontsize=11)
ax.set_ylabel(fy.replace('_', ' ').title(), fontsize=11)
ax.set_title(f'Multivariate Separation: {fx.split("_")[0].title()} vs {fy.split("_")[0].title()}\n'
             f'(point size ~ {fz.replace("_"," ")}, colored by outcome)', fontsize=13, pad=12)

legend = ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95,
                    title='Company Status', title_fontsize=10)
legend.get_frame().set_edgecolor('black')

ax.scatter([safe_pts[fx].median()], [safe_pts[fy].median()], marker='X', s=260,
           c=COLOR_SAFE, edgecolors='black', linewidths=1.5, zorder=5)
ax.scatter([bankrupt_pts[fx].median()], [bankrupt_pts[fy].median()], marker='X', s=260,
           c=COLOR_BANKRUPT, edgecolors='black', linewidths=1.5, zorder=5)

ax.annotate(f'Bankrupt median\n({bankrupt_pts[fx].median():.2f}, {bankrupt_pts[fy].median():.2f})',
            xy=(bankrupt_pts[fx].median(), bankrupt_pts[fy].median()),
            xytext=(0.65, 0.90), textcoords='axes fraction',
            fontsize=9.5, color=COLOR_BANKRUPT, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_BANKRUPT, lw=1.4),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=COLOR_BANKRUPT, alpha=0.9))
ax.annotate(f'Safe median\n({safe_pts[fx].median():.2f}, {safe_pts[fy].median():.2f})',
            xy=(safe_pts[fx].median(), safe_pts[fy].median()),
            xytext=(0.65, 0.78), textcoords='axes fraction',
            fontsize=9.5, color=COLOR_SAFE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_SAFE, lw=1.4),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=COLOR_SAFE, alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUT}/C2_multivariate_scatter.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C2_multivariate_scatter.png")


# -----------------------------------------------------------
# C3. Ratio category-wise grouping
# -----------------------------------------------------------
def categorize(col):
    c = col.lower()
    if any(k in c for k in ['profit', 'ebit', 'ebitda', 'earnings']):
        return 'Profitability'
    if any(k in c for k in ['current_assets', 'cash', 'working_capital', 'liquidit']):
        return 'Liquidity'
    if any(k in c for k in ['liabilit', 'equity', 'capital', 'debt']):
        return 'Leverage / Solvency'
    if any(k in c for k in ['sales', 'inventory', 'receivable', 'turnover', 'rotation', 'cost']):
        return 'Efficiency'
    return 'Company Size'

cat_map = {c: categorize(c) for c in feature_cols}
cat_series = pd.Series(cat_map)

cat_results = []
for cat in cat_series.unique():
    cols_in_cat = cat_series[cat_series == cat].index.tolist()
    pvals = mw_rank.reindex(cols_in_cat).dropna()
    sig_pct = (pvals < 0.05).mean() * 100 if len(pvals) else 0
    cat_results.append({'category': cat, 'n_features': len(cols_in_cat), 'pct_significant': sig_pct})

cat_df = pd.DataFrame(cat_results).sort_values('pct_significant', ascending=False)
reliable = cat_df[cat_df['n_features'] >= 5].sort_values('pct_significant', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6.5))
cat_colors = ['#264653', '#2A9D8F', '#E76F51', '#E9C46A', '#8AB17D']
bars = ax.bar(cat_df['category'], cat_df['pct_significant'],
              color=cat_colors[:len(cat_df)], edgecolor='black', linewidth=0.8, width=0.55)
for bar, val, n in zip(bars, cat_df['pct_significant'], cat_df['n_features']):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.0f}%\n(n={n})',
            ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('% of features in category that significantly differ\nbetween Safe vs Bankrupt (Mann-Whitney p<0.05)', fontsize=10)
ax.set_title('Which Financial Ratio Category Signals Bankruptcy Most Strongly?', fontsize=13, pad=12)
ax.set_ylim(0, 110)
ax.axhline(100, color='gray', linestyle=':', linewidth=1)

top_cat_row = reliable.iloc[0]
top_cat = top_cat_row['category']
top_x_pos = list(cat_df['category']).index(top_cat)
bar_color = cat_colors[top_x_pos % len(cat_colors)]
ax.annotate(f'"{top_cat}" ratios (n={int(top_cat_row["n_features"])}) are the most\nconsistent bankruptcy signal',
            xy=(top_x_pos, top_cat_row['pct_significant']), xytext=(0.42, 0.45), textcoords='axes fraction',
            fontsize=10.5, fontweight='bold', color=bar_color,
            arrowprops=dict(arrowstyle='->', color=bar_color, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=bar_color, alpha=0.95))

plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig(f'{OUT}/C3_category_grouping.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C3_category_grouping.png")


# -----------------------------------------------------------
# C4. Sign-based observation (negative values)
# -----------------------------------------------------------
neg_capable = [c for c in feature_cols if (df[c] < 0).sum() > 0]
neg_series = pd.Series({c: (df[c] < 0).mean() * 100 for c in neg_capable}).sort_values(ascending=False)
top_neg_cols = neg_series.head(10).index.tolist()

neg_pct_safe = [(df.loc[safe_mask, c] < 0).mean() * 100 for c in top_neg_cols]
neg_pct_bankrupt = [(df.loc[bankrupt_mask, c] < 0).mean() * 100 for c in top_neg_cols]
short_names = [c if len(c) <= 28 else c[:25] + '...' for c in top_neg_cols]
x = np.arange(len(top_neg_cols))
width = 0.38

fig, ax = plt.subplots(figsize=(13, 7.5))
bars1 = ax.bar(x - width/2, neg_pct_safe, width, color=COLOR_SAFE, edgecolor='black', linewidth=0.7, label='Safe companies')
bars2 = ax.bar(x + width/2, neg_pct_bankrupt, width, color=COLOR_BANKRUPT, edgecolor='black', linewidth=0.7, label='Bankrupt companies')
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7, f'{b.get_height():.0f}%', ha='center', fontsize=8, color=COLOR_SAFE, fontweight='bold')
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.7, f'{b.get_height():.0f}%', ha='center', fontsize=8, color=COLOR_BANKRUPT, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('% of companies with a NEGATIVE ratio value', fontsize=10.5)
ax.set_title('Which Ratios Turn Negative — and Is It More Common in Bankrupt Companies?', fontsize=13, pad=12)
legend = ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.95)
legend.get_frame().set_edgecolor('black')

biggest_gap_idx = int(np.argmax(np.array(neg_pct_bankrupt) - np.array(neg_pct_safe)))
gap_val = neg_pct_bankrupt[biggest_gap_idx] - neg_pct_safe[biggest_gap_idx]
ax.annotate(f'Largest gap: {short_names[biggest_gap_idx]}\n'
            f'Bankrupt: {neg_pct_bankrupt[biggest_gap_idx]:.0f}%  vs  Safe: {neg_pct_safe[biggest_gap_idx]:.0f}%\n'
            f'(+{gap_val:.0f} pp more likely negative)',
            xy=(biggest_gap_idx + width/2, neg_pct_bankrupt[biggest_gap_idx]),
            xytext=(0.55, 0.75), textcoords='axes fraction', fontsize=9.5, fontweight='bold', color=COLOR_BANKRUPT,
            arrowprops=dict(arrowstyle='->', color=COLOR_BANKRUPT, lw=1.4),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=COLOR_BANKRUPT, alpha=0.95))

plt.tight_layout()
plt.savefig(f'{OUT}/C4_sign_based_observation.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C4_sign_based_observation.png")


# -----------------------------------------------------------
# C5. Inverse-pair consistency check
# -----------------------------------------------------------
col_a = 'total_assets_over_total_liabilities'
col_b = 'total_liabilities_over_total_assets'
pair_df = df[[col_a, col_b]].dropna().copy()
pair_df['product'] = pair_df[col_a] * pair_df[col_b]
consistent_pct = (pair_df['product'].between(0.95, 1.05)).mean() * 100
inconsistent_pct = 100 - consistent_pct

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
full_clip = pair_df['product'].clip(-2, 5)
counts, bins, patches = ax1.hist(full_clip, bins=100, color='#E76F51', edgecolor='white', linewidth=0.2)
for patch, left_edge in zip(patches, bins[:-1]):
    if 0.95 <= left_edge <= 1.05:
        patch.set_facecolor('#2A9D8F')
ax1.axvline(1.0, color='black', linestyle='--', linewidth=1.6)
ax1.set_yscale('log')
ax1.set_xlabel('Product value', fontsize=10.5)
ax1.set_ylabel('Number of companies (log scale)', fontsize=10.5)
ax1.set_title('Full Range (log scale)\nshowing rare violators', fontsize=12)

zoom_df = pair_df[pair_df['product'].between(0.9, 1.1)]
ax2.hist(zoom_df['product'], bins=60, color='#2A9D8F', edgecolor='white', linewidth=0.3)
ax2.axvline(1.0, color='black', linestyle='--', linewidth=1.6)
ax2.set_xlabel('Product value', fontsize=10.5)
ax2.set_ylabel('Number of companies', fontsize=10.5)
ax2.set_title('Zoomed to 0.90-1.10\nthe consistent majority', fontsize=12)

fig.suptitle('Data-Integrity Check: Do Inverse Ratio Pairs Multiply to ~1 as They Should?\n'
             '(total_assets_over_total_liabilities x total_liabilities_over_total_assets)',
             fontsize=14, fontweight='bold', y=1.03)

legend_patches = [
    mpatches.Patch(color='#2A9D8F', label=f'Consistent (0.95-1.05): {consistent_pct:.2f}%'),
    mpatches.Patch(color='#E76F51', label=f'Inconsistent: {inconsistent_pct:.2f}%'),
    plt.Line2D([0], [0], color='black', linestyle='--', label='Theoretical expectation = 1.0')
]
fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10, frameon=True)

fig.text(0.5, -0.04,
          f"Insight: {consistent_pct:.2f}% of rows satisfy the algebraic identity almost exactly — "
          f"the ratio pair is internally consistent, giving confidence the raw values were computed correctly.",
          ha='center', fontsize=10.5, style='italic',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F6F3', edgecolor='#2A9D8F'))

plt.tight_layout()
plt.savefig(f'{OUT}/C5_inverse_pair_consistency.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C5_inverse_pair_consistency.png")
print(f"Inverse-pair consistency: {consistent_pct:.2f}% within [0.95,1.05], {inconsistent_pct:.2f}% inconsistent.")


# -----------------------------------------------------------
# C6. "?" values - specific column check (on RAW data)
# -----------------------------------------------------------
raw_feature_cols = [c for c in raw.columns if c not in ('id', 'forecasting period', 'class')]
qmark_counts = (raw[raw_feature_cols].astype(str) == '?').sum().sort_values(ascending=False)
qmark_counts = qmark_counts[qmark_counts > 0]

if len(qmark_counts) == 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, 'No "?" values found in the input file.\n'
                       '(This file may already be numeric/cleaned —\n'
                       'point INPUT_PATH to the RAW train.csv to see this check properly.)',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#2A9D8F',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F6F3', edgecolor='#2A9D8F'))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('"?" Value Check', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f'{OUT}/C6_questionmark_columns.png', dpi=170, bbox_inches='tight')
    plt.close()
    print("Saved: C6_questionmark_columns.png (no '?' values found — check INPUT_PATH)")
else:
    top_q = qmark_counts.head(15)
    short_q_names = [c if len(c) <= 30 else c[:27] + '...' for c in top_q.index]

    fig, ax = plt.subplots(figsize=(11, 8))
    colors_q = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(top_q)))
    bars = ax.barh(range(len(top_q)), top_q.values[::-1], color=colors_q[::-1], edgecolor='black', linewidth=0.6)
    ax.set_yticks(range(len(top_q)))
    ax.set_yticklabels(short_q_names[::-1], fontsize=9.5)
    ax.set_xlabel('Count of "?" (invalid string) values', fontsize=10.5)
    ax.set_title(f'Where Does the Raw Data Contain "?" Instead of a Number?\n'
                 f'({qmark_counts.sum():,} total "?" cells across {len(qmark_counts)} columns)', fontsize=13, pad=12)
    for i, v in enumerate(top_q.values[::-1]):
        ax.text(v + max(top_q.values)*0.01, i, f'{v:,} ({v/len(raw)*100:.1f}%)', va='center', fontsize=8.5, fontweight='bold')

    ax.annotate('These become NaN after\npd.to_numeric() conversion —\nthis is the ROOT CAUSE of\nmissing values seen earlier',
                xy=(top_q.values[0]*0.5, len(top_q)-1), xytext=(0.45, 0.35), textcoords='axes fraction',
                fontsize=10, fontweight='bold', color='#B5651D',
                arrowprops=dict(arrowstyle='->', color='#B5651D', lw=1.4),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3CD', edgecolor='#B5651D', alpha=0.95))

    plt.tight_layout()
    plt.savefig(f'{OUT}/C6_questionmark_columns.png', dpi=170, bbox_inches='tight')
    plt.close()
    print("Saved: C6_questionmark_columns.png")


# -----------------------------------------------------------
# C7. logarithm_of_total_assets - special observation
# -----------------------------------------------------------
col = 'logarithm_of_total_assets'
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

typical_ratio = 'net_profit_over_total_assets'
lo, hi = df[typical_ratio].quantile([0.02, 0.98])
axes[0].hist(df[col].dropna(), bins=60, color='#457B9D', edgecolor='white', linewidth=0.3,
             density=True, alpha=0.85, label=col.replace('_', ' '))
ax2b = axes[0].twiny()
ax2b.hist(df[typical_ratio].clip(lo, hi).dropna(), bins=60, color='#E76F51', edgecolor='white',
          linewidth=0.3, density=True, alpha=0.45)
axes[0].set_xlabel('logarithm_of_total_assets (already log-scaled)', color='#457B9D', fontsize=9.5)
ax2b.set_xlabel(f'{typical_ratio} (raw ratio, clipped 2-98%)', color='#E76F51', fontsize=9.5)
axes[0].set_ylabel('Density', fontsize=10)
axes[0].set_title('Log-Transformed Feature vs a Typical Raw Ratio\n(notice the near-symmetric bell shape here)', fontsize=11.5)
legend_patches = [mpatches.Patch(color='#457B9D', label='logarithm_of_total_assets'),
                  mpatches.Patch(color='#E76F51', label=typical_ratio + ' (raw)')]
axes[0].legend(handles=legend_patches, loc='upper left', fontsize=8.5, frameon=True)

skew_log = df[col].skew()
skew_raw = df[typical_ratio].skew()
axes[0].annotate(f'Skewness = {skew_log:.2f}\n(vs {skew_raw:.1f} for the raw ratio)',
                  xy=(0.05, 0.85), xycoords='axes fraction', fontsize=9.5, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#457B9D'))

size_bins = pd.qcut(df[col], 5, labels=['Smallest\n20%', '2nd', 'Middle', '4th', 'Largest\n20%'])
size_bankruptcy = df.groupby(size_bins, observed=True)['class'].mean() * 100
bars = axes[1].bar(range(len(size_bankruptcy)), size_bankruptcy.values,
                    color=plt.cm.viridis(np.linspace(0.15, 0.9, len(size_bankruptcy))),
                    edgecolor='black', linewidth=0.8, width=0.6)
axes[1].set_xticks(range(len(size_bankruptcy)))
axes[1].set_xticklabels(size_bankruptcy.index, fontsize=9.5)
axes[1].set_ylabel('Bankruptcy rate (%)', fontsize=10)
axes[1].set_title('Company Size (by log total assets) vs Bankruptcy Rate', fontsize=11.5)
for bar, val in zip(bars, size_bankruptcy.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}%', ha='center', fontsize=9.5, fontweight='bold')

fig.suptitle('The "Odd One Out": logarithm_of_total_assets Is the Only Pre-Transformed Feature',
             fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/C7_log_total_assets_special.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C7_log_total_assets_special.png")


# -----------------------------------------------------------
# C8. `id` column - sequential pattern check
# -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

period_colors = {1: '#264653', 2: '#2A9D8F', 3: '#E9C46A', 4: '#E76F51'}
for period, sub in df.groupby('forecasting period'):
    axes[0].scatter(sub['id'], sub['forecasting period'], s=6, alpha=0.4,
                     color=period_colors.get(period, 'gray'), label=f'Period {int(period)}')
axes[0].set_xlabel('id', fontsize=10.5)
axes[0].set_ylabel('forecasting period', fontsize=10.5)
axes[0].set_title('Is "id" Ordered by Forecasting Period?', fontsize=12)
legend = axes[0].legend(loc='center right', fontsize=9, frameon=True, markerscale=3)
legend.get_frame().set_edgecolor('black')

id_period_corr = df['id'].corr(df['forecasting period'])
axes[0].annotate(f'Correlation(id, period) = {id_period_corr:.3f}\n'
                  f'{"-> near-random ordering" if abs(id_period_corr) < 0.1 else "-> id carries period signal!"}',
                  xy=(0.05, 0.06), xycoords='axes fraction', fontsize=9.5, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black'))

jitter = np.random.normal(0, 0.03, size=len(df))
axes[1].scatter(df['id'], df['class'] + jitter, s=5, alpha=0.15, color=COLOR_SAFE)
axes[1].scatter(df.loc[bankrupt_mask, 'id'], df.loc[bankrupt_mask, 'class'] + jitter[bankrupt_mask.values],
                s=8, alpha=0.5, color=COLOR_BANKRUPT)
axes[1].set_xlabel('id', fontsize=10.5)
axes[1].set_ylabel('class (0=Safe, 1=Bankrupt, jittered)', fontsize=10.5)
axes[1].set_title('Is "id" Correlated with the Target? (Leakage Check)', fontsize=12)

id_class_corr = df['id'].corr(df['class'])
axes[1].annotate(f'Correlation(id, class) = {id_class_corr:.3f}\n'
                  f'{"-> no leakage signal detected" if abs(id_class_corr) < 0.05 else "-> investigate further!"}',
                  xy=(0.05, 0.90), xycoords='axes fraction', fontsize=9.5, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='black'))

fig.suptitle('Sanity Check: Does the "id" Column Hide Any Ordering or Leakage?', fontsize=14, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/C8_id_sequential_pattern.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: C8_id_sequential_pattern.png")

print("\n=== PART C complete (8 advanced EDA steps) ===")
print("\n=== FULL EDA COMPLETE: 17 visuals total (9 original + 8 advanced) ===")


# D1. Top correlated feature-pairs — actual scatter plots
# -----------------------------------------------------------
corr_matrix = df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
pairs = upper.stack().sort_values(ascending=False)
pairs = pairs[pairs < 0.999]  # drop near-duplicate/identical columns
top_pairs = pairs.head(3)

fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
fig.suptitle('Top 3 Most Correlated Feature Pairs — Actual Relationship', fontsize=15, fontweight='bold', y=1.03)
pair_colors = ['#264653', '#2A9D8F', '#E76F51']

for ax, ((cx, cy), corr_val), pc in zip(axes, top_pairs.items(), pair_colors):
    lo_x, hi_x = df[cx].quantile([0.02, 0.98])
    lo_y, hi_y = df[cy].quantile([0.02, 0.98])
    pdf = df[[cx, cy, 'class']].copy()
    pdf[cx] = pdf[cx].clip(lo_x, hi_x)
    pdf[cy] = pdf[cy].clip(lo_y, hi_y)

    ax.scatter(pdf.loc[safe_mask, cx], pdf.loc[safe_mask, cy], s=14, alpha=0.35,
               color=COLOR_SAFE, edgecolors='none', label='Safe')
    ax.scatter(pdf.loc[bankrupt_mask, cx], pdf.loc[bankrupt_mask, cy], s=26, alpha=0.75,
               color=COLOR_BANKRUPT, edgecolors='black', linewidths=0.3, label='Bankrupt')

    short_x = cx if len(cx) <= 26 else cx[:23] + '...'
    short_y = cy if len(cy) <= 26 else cy[:23] + '...'
    ax.set_xlabel(short_x, fontsize=8.5)
    ax.set_ylabel(short_y, fontsize=8.5)
    ax.set_title(f'r = {corr_val:.3f}', fontsize=12, fontweight='bold', color=pc)
    for spine in ax.spines.values():
        spine.set_edgecolor(pc)
        spine.set_linewidth(1.4)
    leg = ax.legend(fontsize=8.5, loc='upper left', frameon=True)
    leg.get_frame().set_edgecolor('black')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{OUT}/D1_top_correlated_pairs_scatter.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: D1_top_correlated_pairs_scatter.png")
print("Top correlated pairs:\n", top_pairs)


# -----------------------------------------------------------
# D2. Altman Z-Score benchmark
# -----------------------------------------------------------
# Classic Altman (1968) Z-Score:
#   Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
#   X1 = working_capital / total_assets
#   X2 = retained_earnings / total_assets
#   X3 = EBIT / total_assets
#   X4 = market_value_of_equity / total_liabilities  -> no market value available,
#        proxied here with book_value_of_equity_over_total_liabilities
#   X5 = sales / total_assets
z_df = df.copy()
z_df['altman_z'] = (
    1.2 * z_df['working_capital_over_total_assets'] +
    1.4 * z_df['retained_earnings_over_total_assets'] +
    3.3 * z_df['EBIT_over_total_assets'] +
    0.6 * z_df['book_value_of_equity_over_total_liabilities'] +
    1.0 * z_df['sales_over_total_assets']
)

lo, hi = z_df['altman_z'].quantile([0.01, 0.99])
z_clip = z_df['altman_z'].clip(lo, hi)

fig, ax = plt.subplots(figsize=(12, 7))
ax.axvspan(lo, 1.81, color="#E76F51", alpha=0.15, label='Distress zone (Z < 1.81)')
ax.axvspan(1.81, 2.99, color='#E9C46A', alpha=0.20, label='Grey zone (1.81–2.99)')
ax.axvspan(2.99, hi, color='#2A9D8F', alpha=0.15, label='Safe zone (Z > 2.99)')

ax.hist(z_clip[safe_mask], bins=80, alpha=0.65, color=COLOR_SAFE, density=True,
        edgecolor='white', linewidth=0.3, label='Safe (actual)')
ax.hist(z_clip[bankrupt_mask], bins=80, alpha=0.7, color=COLOR_BANKRUPT, density=True,
        edgecolor='white', linewidth=0.3, label='Bankrupt (actual)')
ax.axvline(1.81, color='black', linestyle='--', linewidth=1.2)
ax.axvline(2.99, color='black', linestyle='--', linewidth=1.2)

ax.set_xlabel('Altman Z-Score', fontsize=10.5)
ax.set_ylabel('Density', fontsize=10.5)
ax.set_title('Altman Z-Score Benchmark vs Actual Bankruptcy Status', fontsize=13, pad=12)
legend = ax.legend(fontsize=9, loc='upper right', frameon=True)
legend.get_frame().set_edgecolor('black')

pct_bankrupt_in_distress = (z_df.loc[bankrupt_mask, 'altman_z'] < 1.81).mean() * 100
pct_safe_in_safe_zone = (z_df.loc[safe_mask, 'altman_z'] > 2.99).mean() * 100
ax.annotate(f'{pct_bankrupt_in_distress:.1f}% of actually-bankrupt firms\nfall in the distress zone\n\n'
            f'{pct_safe_in_safe_zone:.1f}% of actually-safe firms\nfall in the safe zone',
            xy=(0.02, 0.62), xycoords='axes fraction', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUT}/D2_altman_zscore_benchmark.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: D2_altman_zscore_benchmark.png")
print(f"Altman Z-Score: {pct_bankrupt_in_distress:.2f}% of bankrupt firms correctly flagged as distressed, "
      f"{pct_safe_in_safe_zone:.2f}% of safe firms correctly flagged as safe.")


# -----------------------------------------------------------
# D3. PCA visualization (65 features -> 2D)
# -----------------------------------------------------------
X_scaled = StandardScaler().fit_transform(df[feature_cols])
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
pca_df['class'] = df['class'].values

fig, ax = plt.subplots(figsize=(10.5, 8.5))
ax.scatter(pca_df.loc[pca_df['class'] == 0, 'PC1'], pca_df.loc[pca_df['class'] == 0, 'PC2'],
           s=14, alpha=0.35, color=COLOR_SAFE, edgecolors='none', label=f'Safe (n={safe_mask.sum():,})')
ax.scatter(pca_df.loc[pca_df['class'] == 1, 'PC1'], pca_df.loc[pca_df['class'] == 1, 'PC2'],
           s=26, alpha=0.75, color=COLOR_BANKRUPT, edgecolors='black', linewidths=0.3,
           label=f'Bankrupt (n={bankrupt_mask.sum():,})')

var1, var2 = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f'PC1 ({var1:.1f}% variance explained)', fontsize=10.5)
ax.set_ylabel(f'PC2 ({var2:.1f}% variance explained)', fontsize=10.5)
ax.set_title(f'PCA Projection of All {len(feature_cols)} Features (2D)\n'
             f'Total variance captured: {var1+var2:.1f}%', fontsize=13, pad=12)
legend = ax.legend(fontsize=10, loc='upper right', frameon=True, title='Company Status', title_fontsize=9.5)
legend.get_frame().set_edgecolor('black')

ax.annotate('Little visual separation on PC1/PC2 alone\n-> bankruptcy signal is likely spread\nacross many components, not 1-2',
            xy=(0.03, 0.03), xycoords='axes fraction', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3CD', edgecolor='#E9C46A'))

plt.tight_layout()
plt.savefig(f'{OUT}/D3_pca_2d_projection.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: D3_pca_2d_projection.png")
print(f"PCA explained variance: PC1={var1:.2f}%, PC2={var2:.2f}%, total={var1+var2:.2f}%")





# Rebuild summaries here so order doesn't matter
altman_summary = pd.DataFrame({
    'metric': ['pct_bankrupt_correctly_flagged_distress', 'pct_safe_correctly_flagged_safe'],
    'value': [pct_bankrupt_in_distress, pct_safe_in_safe_zone]
})



# ===========================================================
# SAVE ALL EDA TABLES IN ONE EXCEL FILE
# ===========================================================
PROCESSED_OUT = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(PROCESSED_OUT, exist_ok=True)

excel_path = f'{PROCESSED_OUT}/eda_summary.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    missingness_test.to_excel(writer, sheet_name='missingness', index=False)
    corr_with_target.reset_index().rename(columns={'index':'feature',0:'abs_correlation'}).to_excel(writer, sheet_name='correlation', index=False)
    mw_df.to_excel(writer, sheet_name='mannwhitney', index=False)
    norm_df.to_excel(writer, sheet_name='shapiro', index=False)
    skew_kurt.reset_index().rename(columns={'index':'feature'}).to_excel(writer, sheet_name='skew_kurtosis', index=False)
    violations_df.reset_index().rename(columns={'index':'feature'}).to_excel(writer, sheet_name='bounded_violations', index=False)
    cat_df.to_excel(writer, sheet_name='category_significance', index=False)
    top_pairs.reset_index().rename(columns={'level_0':'feature_1','level_1':'feature_2',0:'correlation'}).to_excel(writer, sheet_name='correlated_pairs', index=False)
    altman_summary.to_excel(writer, sheet_name='altman_zscore', index=False)
 

print(f"Saved all EDA tables to: {excel_path}")
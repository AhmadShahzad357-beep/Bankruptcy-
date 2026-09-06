"""
08 - SHAP Explainability (Tuned XGBoost)
==========================================
Loads the Optuna-tuned XGBoost model and explains its predictions:
  1. Global feature importance (mean |SHAP|) - custom bar chart
  2. Beeswarm summary plot - shows direction (increases/decreases risk)
  3. Dependence plots for top 3 features
  4. Waterfall plots for one correctly-flagged bankrupt case
     and one correctly-flagged safe case
  5. Decision plot - how predictions build up across features
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib

plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

COLOR_RISK_UP = '#E63946'    # feature pushes prediction toward BANKRUPT
COLOR_RISK_DOWN = '#2E86AB'  # feature pushes prediction toward SAFE
COLOR_BAR = '#8E44AD'        # tuned-model purple, consistent with script 06

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features_final.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost_tuned.pkl')
OUT = os.path.join(PROJECT_ROOT, 'reports', 'figures')
REPORTS_OUT = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(OUT, exist_ok=True)

# ===========================================================
# 1. Load model + test data
# ===========================================================
model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)
test_df = df[df['split'] == 'test'].drop(columns=['split'])
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

test_proba = model.predict_proba(X_test)[:, 1]
print(f"Loaded tuned model and test set: {X_test.shape}")

# ===========================================================
# 2. Compute SHAP values (TreeExplainer - exact & fast for XGBoost)
# ===========================================================
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
expected_value = explainer.expected_value
print("SHAP values computed for test set.")

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X_test.columns)
mean_signed_shap = pd.Series(shap_values.mean(axis=0), index=X_test.columns)

# ===========================================================
# 3. Global importance - custom bar chart (mean |SHAP|, top 15)
# ===========================================================
top15 = mean_abs_shap.sort_values(ascending=False).head(15)
top15_direction = mean_signed_shap.reindex(top15.index)
bar_colors = [COLOR_RISK_UP if v > 0 else COLOR_RISK_DOWN for v in top15_direction]

fig, ax = plt.subplots(figsize=(10.5, 8))
labels = [c if len(c) <= 32 else c[:29] + '...' for c in top15.index[::-1]]
ax.barh(range(len(top15)), top15.values[::-1], color=bar_colors[::-1],
        edgecolor='black', linewidth=0.7, alpha=0.9)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel('Mean |SHAP value|  (average impact on model output)', fontsize=10.5)
ax.set_title('Global Feature Importance — Top 15 (SHAP)', fontsize=13.5, pad=12)
legend_handles = [
    mpatches.Patch(facecolor=COLOR_RISK_UP, edgecolor='black', label='On average, raises bankruptcy risk'),
    mpatches.Patch(facecolor=COLOR_RISK_DOWN, edgecolor='black', label='On average, lowers bankruptcy risk')
]
legend = ax.legend(handles=legend_handles, fontsize=9.5, loc='lower right', frameon=True)
legend.get_frame().set_edgecolor('black')
top_feat = top15.index[0]
ax.annotate(f'Most influential feature:\n{top_feat[:35]}',
            xy=(top15.values[0], len(top15)-1), xytext=(0.35, 0.08), textcoords='axes fraction',
            fontsize=9, fontweight='bold', color=COLOR_BAR,
            arrowprops=dict(arrowstyle='->', color=COLOR_BAR, lw=1.3),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=COLOR_BAR))
plt.tight_layout()
plt.savefig(f'{OUT}/S1_global_importance.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: S1_global_importance.png")

# ===========================================================
# 4. Beeswarm summary plot (direction + magnitude per data point)
# ===========================================================
fig = plt.figure(figsize=(10.5, 8.5))
shap.summary_plot(shap_values, X_test, show=False, max_display=15, plot_size=None)
fig = plt.gcf()
fig.set_size_inches(10.5, 8.5)
ax = plt.gca()
ax.set_title('SHAP Summary — Feature Value vs Impact on Prediction\n'
             '(red = high feature value, blue = low feature value)', fontsize=12.5, pad=14)
ax.set_xlabel('SHAP value (impact on predicted bankruptcy probability)', fontsize=10.5)
plt.tight_layout()
plt.savefig(f'{OUT}/S2_beeswarm_summary.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: S2_beeswarm_summary.png")

# ===========================================================
# 5. Dependence plots for top 3 features
# ===========================================================
top3_features = top15.index[:3].tolist()
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
dep_colors = ['#264653', '#2A9D8F', '#E76F51']

for ax, feat, dcolor in zip(axes, top3_features, dep_colors):
    feat_idx = X_test.columns.get_loc(feat)
    x_vals = X_test[feat].values
    y_vals = shap_values[:, feat_idx]
    lo, hi = np.percentile(x_vals, [1, 99])
    mask = (x_vals >= lo) & (x_vals <= hi)
    sc = ax.scatter(x_vals[mask], y_vals[mask], c=y_test[mask], cmap='coolwarm',
                     s=16, alpha=0.6, edgecolors='none')
    ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.6)
    short_name = feat if len(feat) <= 30 else feat[:27] + '...'
    ax.set_xlabel(short_name, fontsize=9)
    ax.set_ylabel('SHAP value', fontsize=9.5)
    ax.set_title(f'Dependence: {short_name}', fontsize=11, color=dcolor)

cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
cbar.set_label('Actual class (0=Safe, 1=Bankrupt)', fontsize=9.5)
fig.suptitle('SHAP Dependence Plots — Top 3 Most Important Features', fontsize=14, fontweight='bold', y=1.03)
plt.savefig(f'{OUT}/S3_dependence_plots.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: S3_dependence_plots.png")

# ===========================================================
# 6. Waterfall plots — one bankrupt case, one safe case
#    (correctly predicted, highest-confidence example of each)
# ===========================================================
bankrupt_mask = (y_test.values == 1) & (test_proba >= 0.5)
safe_mask = (y_test.values == 0) & (test_proba < 0.5)

bankrupt_idx = np.where(bankrupt_mask)[0]
safe_idx = np.where(safe_mask)[0]

example_bankrupt = bankrupt_idx[np.argmax(test_proba[bankrupt_idx])] if len(bankrupt_idx) else None
example_safe = safe_idx[np.argmin(test_proba[safe_idx])] if len(safe_idx) else None

for idx, label, fname in [(example_bankrupt, 'Bankrupt', 'S4_waterfall_bankrupt.png'),
                           (example_safe, 'Safe', 'S5_waterfall_safe.png')]:
    if idx is None:
        continue
    exp = shap.Explanation(values=shap_values[idx], base_values=expected_value,
                            data=X_test.iloc[idx].values, feature_names=X_test.columns.tolist())
    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.suptitle(f'SHAP Waterfall — Correctly Predicted "{label}" Company\n'
                 f'(predicted probability = {test_proba[idx]:.3f})', fontsize=12.5, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{OUT}/{fname}', dpi=170, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

# ===========================================================
# 7. Decision plot — how predictions build up (sample of 20 companies)
# ===========================================================
np.random.seed(42)
sample_size = 20
sample_idx = np.random.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)

fig = plt.figure(figsize=(11, 8.5))
shap.decision_plot(expected_value, shap_values[sample_idx], X_test.iloc[sample_idx],
                    feature_display_range=slice(None, -16, -1), show=False,
                    highlight=list(np.where(y_test.values[sample_idx] == 1)[0]))
fig = plt.gcf()
fig.suptitle('SHAP Decision Plot — 20 Sample Companies\n(highlighted lines = actually bankrupt)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/S6_decision_plot.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: S6_decision_plot.png")

# ===========================================================
# 8. Save SHAP importance table (CSV)
# ===========================================================
shap_summary_df = pd.DataFrame({
    'feature': mean_abs_shap.index,
    'mean_abs_shap': mean_abs_shap.values,
    'mean_signed_shap': mean_signed_shap.reindex(mean_abs_shap.index).values
}).sort_values('mean_abs_shap', ascending=False)

shap_summary_df.to_csv(f'{REPORTS_OUT}/shap_feature_importance.csv', index=False)
print(f"Saved: {REPORTS_OUT}/shap_feature_importance.csv")

print("\n=== SHAP explainability complete: 6 graphs (S1-S6) + 1 CSV ===")
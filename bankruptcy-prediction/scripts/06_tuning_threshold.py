"""
06 - Hyperparameter Tuning with Optuna (XGBoost only)
==========================================
- 50 trials, no timeout (runs until all 50 complete)
- 5-fold Stratified CV (scoring: PR-AUC)
- n_estimators capped at 400 (was 600) - reduced for speed
- Uses X_train (train-fit portion) only - validation/test untouched
- Saves: optimization history, param importance, before/after comparison,
         tuned model (pkl, in models/), tuning summary (csv, in data/processed/),
         best hyperparameters (json, in reports/)
- NEW: evaluates the tuned model on the held-out VALIDATION set (threshold
  tuning) and then the TEST set (final report), and compares it against the
  baseline model saved by 05_model_training.py.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import optuna
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
import xgboost as xgb
import joblib

optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

COLOR_XGB = '#2E86AB'        # baseline blue
COLOR_TUNED = '#8E44AD'      # tuned purple (distinct from baseline blue/orange used elsewhere)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features_final.csv')
OUT = os.path.join(PROJECT_ROOT, 'reports', 'figures')
MODELS_OUT = os.path.join(PROJECT_ROOT, 'models')
REPORTS_OUT = os.path.join(PROJECT_ROOT, 'reports')
PROCESSED_OUT = os.path.join(PROJECT_ROOT, 'data', 'processed')
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODELS_OUT, exist_ok=True)
os.makedirs(PROCESSED_OUT, exist_ok=True)

# ===========================================================
# 1. Load data (train-fit portion only, same split as 05)
# ===========================================================
df = pd.read_csv(DATA_PATH)
train_df = df[df['split'] == 'train'].drop(columns=['split'])
test_df = df[df['split'] == 'test'].drop(columns=['split'])
X_train_full = train_df.drop(columns=['class'])
y_train_full = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 5-fold CV for a more reliable estimate per trial (takes longer than 3-fold,
# but the score used to pick the best hyperparameters is more stable).
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Tuning on train-fit set: {X_train.shape}, scale_pos_weight={scale_pos_weight:.3f}")
print("Using 5-fold CV and n_estimators capped at 400 for faster tuning.")

# ===========================================================
# 2. Baseline (untuned) CV score
# ===========================================================
xgb_base = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric='logloss',
    tree_method='hist', random_state=42, n_jobs=-1
)
xgb_base_scores = cross_val_score(xgb_base, X_train, y_train, cv=cv, scoring='average_precision')
print(f"Baseline XGBoost CV PR-AUC: {xgb_base_scores.mean():.4f}")

# ===========================================================
# 3. Optuna objective function
# ===========================================================
def xgb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),   # was 100-600
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='average_precision')
    return scores.mean()

# ===========================================================
# 4. Run Optuna study (50 trials, no timeout - runs until complete)
# ===========================================================
print("\nTuning XGBoost (50 trials, no time limit)...")
xgb_study = optuna.create_study(direction='maximize', study_name='xgb_tuning',
                                 sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=50, show_progress_bar=True)

print(f"\nTrials completed: {len(xgb_study.trials)}")
print(f"Best XGBoost CV PR-AUC: {xgb_study.best_value:.4f}")
print(f"Best XGBoost params: {xgb_study.best_params}")

# ===========================================================
# 5. Optimization history plot
# ===========================================================
fig, ax = plt.subplots(figsize=(9.5, 6.5))
trial_vals = [t.value for t in xgb_study.trials]
best_so_far = np.maximum.accumulate(trial_vals)
ax.scatter(range(len(trial_vals)), trial_vals, s=20, alpha=0.4, color=COLOR_XGB, label='Trial score')
ax.plot(range(len(trial_vals)), best_so_far, color='black', linewidth=2, label='Best so far')
ax.axhline(best_so_far[-1], color=COLOR_XGB, linestyle='--', linewidth=1.2, alpha=0.7)
ax.set_xlabel('Trial number', fontsize=10.5)
ax.set_ylabel('PR-AUC (5-fold CV)', fontsize=10.5)
ax.set_title(f'XGBoost — Optuna Optimization History ({len(trial_vals)} trials)', fontsize=13, pad=12)
ax.annotate(f'Best: {best_so_far[-1]:.4f}', xy=(len(trial_vals)-1, best_so_far[-1]),
            xytext=(0.55, 0.15), textcoords='axes fraction', fontsize=10.5, fontweight='bold',
            color=COLOR_XGB, arrowprops=dict(arrowstyle='->', color=COLOR_XGB, lw=1.3),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=COLOR_XGB))
legend = ax.legend(fontsize=9.5, loc='lower right', frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/T1_optimization_history.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T1_optimization_history.png")

# ===========================================================
# 6. Hyperparameter importance plot
# ===========================================================
fig, ax = plt.subplots(figsize=(9.5, 7))
importance = optuna.importance.get_param_importances(xgb_study)
params = list(importance.keys())
values = list(importance.values())
ax.barh(range(len(params)), values[::-1], color=COLOR_XGB, edgecolor='black', linewidth=0.7, alpha=0.88)
ax.set_yticks(range(len(params)))
ax.set_yticklabels(params[::-1], fontsize=10)
ax.set_xlabel('Relative importance', fontsize=10.5)
ax.set_title('XGBoost — Hyperparameter Importance', fontsize=13, pad=12)
plt.tight_layout()
plt.savefig(f'{OUT}/T2_hyperparam_importance.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T2_hyperparam_importance.png")

# ===========================================================
# 7. Before vs After tuning comparison (CV score, not test set)
# ===========================================================
fig, ax = plt.subplots(figsize=(7, 6.5))
before = xgb_base_scores.mean()
before_std = xgb_base_scores.std()
after = xgb_study.best_value

bars = ax.bar(['Before tuning\n(default params)', 'After tuning\n(Optuna)'],
              [before, after], yerr=[before_std, 0], capsize=6,
              color=['#B0B0B0', COLOR_XGB], edgecolor='black', linewidth=0.9, width=0.55)
for bar, val in zip(bars, [before, after]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.008, f'{val:.4f}',
            ha='center', fontsize=11, fontweight='bold')
gain = ((after - before) / before) * 100
ax.annotate(f'{"+" if gain >= 0 else ""}{gain:.1f}% improvement',
            xy=(0.5, max(before, after) + 0.035), xycoords=('axes fraction', 'data'),
            fontsize=11.5, fontweight='bold', ha='center',
            color='#2A9D8F' if gain >= 0 else '#E63946')
ax.set_ylabel('PR-AUC (5-fold CV mean)', fontsize=10.5)
ax.set_title('XGBoost: Before vs After Hyperparameter Tuning\n(cross-validation score, not test set)', fontsize=12.5, pad=14)
plt.tight_layout()
plt.savefig(f'{OUT}/T3_before_after_tuning.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T3_before_after_tuning.png")

# ===========================================================
# 8. Train final tuned model on train-fit, save everything
# ===========================================================
xgb_best_params = {**xgb_study.best_params, 'scale_pos_weight': scale_pos_weight,
                    'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42, 'n_jobs': -1}

xgb_tuned = xgb.XGBClassifier(**xgb_best_params)
xgb_tuned.fit(X_train, y_train)

# Model (.pkl) -> models/
joblib.dump(xgb_tuned, f'{MODELS_OUT}/xgboost_tuned.pkl')

# Hyperparameters (.json) -> reports/  (not a CSV/data file, so stays with other reports)
with open(f'{REPORTS_OUT}/best_hyperparameters.json', 'w') as f:
    json.dump({'xgboost': xgb_study.best_params}, f, indent=2)

# Tuning summary (.csv) -> data/processed/  (per your instruction: CSVs belong in data/processed)
tuning_summary = pd.DataFrame([
    {'model': 'XGBoost', 'cv_folds': 5, 'n_trials': len(xgb_study.trials),
     'cv_pr_auc_before': before, 'cv_pr_auc_after': after,
     'improvement_pct': round(gain, 2)},
])
tuning_summary.to_csv(f'{PROCESSED_OUT}/tuning_summary.csv', index=False)

print(f"\nSaved tuned model to: {MODELS_OUT}/xgboost_tuned.pkl")
print(f"Saved best hyperparameters to: {REPORTS_OUT}/best_hyperparameters.json")
print(f"Saved tuning summary to: {PROCESSED_OUT}/tuning_summary.csv")
print(f"\n=== Optuna tuning complete (XGBoost only): {len(xgb_study.trials)} trials, "
      f"3 graphs + tuned model + hyperparameter JSON + summary CSV ===")


# ===========================================================
# 9. TUNED MODEL — threshold tuning on VALIDATION (leakage-safe)
#    then final report on TEST (only touched once, at the end)
# ===========================================================
tuned_val_proba = xgb_tuned.predict_proba(X_val)[:, 1]
tuned_test_proba = xgb_tuned.predict_proba(X_test)[:, 1]

precision_v, recall_v, thresholds_v = precision_recall_curve(y_val, tuned_val_proba)
f1_v = 2 * precision_v * recall_v / (precision_v + recall_v + 1e-10)
best_idx = np.argmax(f1_v[:-1])
tuned_best_thresh = thresholds_v[best_idx]
print(f"\nTuned XGBoost — best threshold (chosen on validation): {tuned_best_thresh:.3f}")

# --- Threshold tuning curve for the TUNED model ---
fig, ax = plt.subplots(figsize=(8.5, 6.5))
ax.plot(thresholds_v, precision_v[:-1], color='#264653', linewidth=1.8, label='Precision')
ax.plot(thresholds_v, recall_v[:-1], color='#2A9D8F', linewidth=1.8, label='Recall')
ax.plot(thresholds_v, f1_v[:-1], color=COLOR_TUNED, linewidth=2.6, label='F1-score')
ax.axvline(tuned_best_thresh, color='black', linestyle='--', linewidth=1.3)
ax.scatter([tuned_best_thresh], [f1_v[best_idx]], color=COLOR_TUNED, s=110, zorder=5,
           edgecolors='black', linewidths=1.2)
ax.annotate(f'Best threshold = {tuned_best_thresh:.3f}\nF1 (on validation) = {f1_v[best_idx]:.3f}',
            xy=(tuned_best_thresh, f1_v[best_idx]), xytext=(0.42, 0.15), textcoords='axes fraction',
            fontsize=9.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=1.3, color=COLOR_TUNED),
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=COLOR_TUNED))
ax.set_xlabel('Threshold', fontsize=10.5)
ax.set_ylabel('Score', fontsize=10.5)
ax.set_title('Tuned XGBoost: Threshold Tuning (on VALIDATION set)', fontsize=13, pad=12)
legend = ax.legend(fontsize=9.5, loc='center left', frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/T4_tuned_threshold_tuning.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T4_tuned_threshold_tuning.png")

# ===========================================================
# 10. Load BASELINE model (from 05) for a fair side-by-side comparison
# ===========================================================
baseline_path = f'{MODELS_OUT}/xgboost_baseline.pkl'
have_baseline = os.path.exists(baseline_path)

if have_baseline:
    xgb_baseline = joblib.load(baseline_path)
    base_val_proba = xgb_baseline.predict_proba(X_val)[:, 1]
    base_test_proba = xgb_baseline.predict_proba(X_test)[:, 1]

    precision_b, recall_b, thresholds_b = precision_recall_curve(y_val, base_val_proba)
    f1_b = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-10)
    base_best_idx = np.argmax(f1_b[:-1])
    base_best_thresh = thresholds_b[base_best_idx]
    print(f"Baseline XGBoost — best threshold (chosen on validation): {base_best_thresh:.3f}")
else:
    print("\n(No baseline model found at models/xgboost_baseline.pkl — "
          "run 05_model_training.py first for a full comparison. Continuing with tuned-model-only report.)")

# ===========================================================
# 11. ROC + PR curves on TEST — baseline vs tuned
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(15.5, 7))

# ROC
ax = axes[0]
if have_baseline:
    fpr_b, tpr_b, _ = roc_curve(y_test, base_test_proba)
    auc_b = roc_auc_score(y_test, base_test_proba)
    ax.plot(fpr_b, tpr_b, color=COLOR_XGB, linewidth=2.2, label=f'Baseline (AUC={auc_b:.3f})')
fpr_t, tpr_t, _ = roc_curve(y_test, tuned_test_proba)
auc_t = roc_auc_score(y_test, tuned_test_proba)
ax.plot(fpr_t, tpr_t, color=COLOR_TUNED, linewidth=2.2, label=f'Tuned (AUC={auc_t:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.1, label='Random guess')
ax.set_xlabel('False Positive Rate', fontsize=10.5)
ax.set_ylabel('True Positive Rate', fontsize=10.5)
ax.set_title('ROC Curve — Baseline vs Tuned (Test Set)', fontsize=12.5)
legend = ax.legend(fontsize=9.5, loc='lower right', frameon=True)
legend.get_frame().set_edgecolor('black')

# PR
ax = axes[1]
if have_baseline:
    prec_b, rec_b, _ = precision_recall_curve(y_test, base_test_proba)
    ap_b = average_precision_score(y_test, base_test_proba)
    ax.plot(rec_b, prec_b, color=COLOR_XGB, linewidth=2.2, label=f'Baseline (AP={ap_b:.3f})')
prec_t, rec_t, _ = precision_recall_curve(y_test, tuned_test_proba)
ap_t = average_precision_score(y_test, tuned_test_proba)
ax.plot(rec_t, prec_t, color=COLOR_TUNED, linewidth=2.2, label=f'Tuned (AP={ap_t:.3f})')
ax.axhline(y_test.mean(), color='gray', linestyle='--', linewidth=1.1, label=f'Baseline rate ({y_test.mean():.3f})')
ax.set_xlabel('Recall', fontsize=10.5)
ax.set_ylabel('Precision', fontsize=10.5)
ax.set_title('Precision-Recall Curve — Baseline vs Tuned (Test Set)', fontsize=12.5)
legend = ax.legend(fontsize=9.5, loc='upper right', frameon=True)
legend.get_frame().set_edgecolor('black')

fig.suptitle('XGBoost Baseline vs Optuna-Tuned — Test Set Ranking Quality', fontsize=14.5, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/T5_baseline_vs_tuned_roc_pr.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T5_baseline_vs_tuned_roc_pr.png")

# ===========================================================
# 12. Confusion matrix — tuned model on TEST at its tuned threshold
# ===========================================================
fig, ax = plt.subplots(figsize=(6.5, 6))
preds_tuned = (tuned_test_proba >= tuned_best_thresh).astype(int)
cm = confusion_matrix(y_test, preds_tuned)
im = ax.imshow(cm, cmap='Purples')
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=15, fontweight='bold',
                color='white' if cm[i, j] > cm.max()/2 else 'black')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Safe', 'Bankrupt']); ax.set_yticklabels(['Safe', 'Bankrupt'])
ax.set_xlabel('Predicted', fontsize=10.5); ax.set_ylabel('Actual', fontsize=10.5)
ax.set_title(f'Tuned XGBoost — Test Set\n(threshold={tuned_best_thresh:.3f}, from validation)',
             fontsize=12, color=COLOR_TUNED)
plt.tight_layout()
plt.savefig(f'{OUT}/T6_tuned_confusion_matrix.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T6_tuned_confusion_matrix.png")

# ===========================================================
# 13. Final comparison table — baseline vs tuned, on TEST set
# ===========================================================
final_rows = []
if have_baseline:
    for label, proba_v, thresh_v, proba_t in [
        ('Baseline', base_val_proba, base_best_thresh, base_test_proba),
    ]:
        preds = (proba_t >= thresh_v).astype(int)
        final_rows.append({
            'model': label, 'threshold': round(float(thresh_v), 3),
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'precision': round(precision_score(y_test, preds), 4),
            'recall': round(recall_score(y_test, preds), 4),
            'f1': round(f1_score(y_test, preds), 4),
            'roc_auc': round(roc_auc_score(y_test, proba_t), 4),
            'pr_auc': round(average_precision_score(y_test, proba_t), 4),
        })

preds_tuned_final = (tuned_test_proba >= tuned_best_thresh).astype(int)
final_rows.append({
    'model': 'Tuned (Optuna)', 'threshold': round(float(tuned_best_thresh), 3),
    'accuracy': round(accuracy_score(y_test, preds_tuned_final), 4),
    'precision': round(precision_score(y_test, preds_tuned_final), 4),
    'recall': round(recall_score(y_test, preds_tuned_final), 4),
    'f1': round(f1_score(y_test, preds_tuned_final), 4),
    'roc_auc': round(roc_auc_score(y_test, tuned_test_proba), 4),
    'pr_auc': round(average_precision_score(y_test, tuned_test_proba), 4),
})

final_comparison_df = pd.DataFrame(final_rows)
print("\n=== FINAL: Baseline vs Tuned XGBoost — Test Set (threshold from validation, no leakage) ===")
print(final_comparison_df)

# --- Table as image ---
fig, ax = plt.subplots(figsize=(11.5, 2.6))
ax.axis('off')
tbl_cols = ['model', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
cell_text = final_comparison_df[tbl_cols].values
row_colors = ['#EAF3F8' if r == 'Baseline' else '#F1E9F7' for r in final_comparison_df['model']]
table = ax.table(cellText=cell_text, colLabels=tbl_cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.1)
for i in range(len(cell_text)):
    for j in range(len(tbl_cols)):
        table[i+1, j].set_facecolor(row_colors[i])
        table[i+1, j].set_edgecolor('black')
for j in range(len(tbl_cols)):
    table[0, j].set_facecolor('#264653')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_edgecolor('black')
ax.set_title('Final Comparison — XGBoost Baseline vs Optuna-Tuned (Test Set)', fontsize=12.5, fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig(f'{OUT}/T7_final_comparison_table.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: T7_final_comparison_table.png")

# ===========================================================
# 14. Save everything (CSV + tuned test predictions)
# ===========================================================
final_comparison_df.to_csv(f'{REPORTS_OUT}/tuned_vs_baseline_test_comparison.csv', index=False)
print(f"Saved: {REPORTS_OUT}/tuned_vs_baseline_test_comparison.csv")

tuned_predictions_df = pd.DataFrame({
    'actual_class': y_test.values,
    'tuned_probability': tuned_test_proba,
    'tuned_predicted': preds_tuned_final,
}, index=X_test.index)
tuned_predictions_df.to_csv(f'{REPORTS_OUT}/tuned_test_predictions.csv', index=True)
print(f"Saved: {REPORTS_OUT}/tuned_test_predictions.csv")

print("\n=== Tuned-model test evaluation complete: 4 more graphs (T4-T7) + 2 CSVs ===")
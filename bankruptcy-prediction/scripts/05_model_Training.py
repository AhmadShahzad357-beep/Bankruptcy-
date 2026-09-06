"""
05 - Model Training (XGBoost + CatBoost)
==========================================
Order:
  1. Load features_final.csv, split back into train/test
  2. Carve a VALIDATION split out of train (for threshold tuning - avoids leakage)
  3. Compute class-weight (scale_pos_weight) from TRAIN-FIT only
  4. Stratified 5-fold CV for both models -> CV score graph
  5. Train final XGBoost + CatBoost on the train-fit portion
  6. ROC curve + PR curve on the held-out TEST set
  7. Threshold tuning on VALIDATION set only (never touches test)
  8. Confusion matrices on TEST set at default (0.5) and tuned threshold
  9. Feature importance (both models)
  10. Final comparison table

LEAKAGE FIX vs the previous version:
  The optimal decision threshold is now chosen using a held-out
  VALIDATION split carved out of the training data - never from the
  test set. The test set is only ever used once, at the end, to report
  final numbers at that already-chosen threshold. This mirrors what
  would happen with genuinely unseen future data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score
)
import xgboost as xgb
from catboost import CatBoostClassifier

# ---------------- Global style ----------------
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 10.5
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 13.5
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

COLOR_XGB = '#2E86AB'    # blue
COLOR_CAT = '#E76F51'    # orange
COLOR_SAFE = '#2A9D8F'
COLOR_BANKRUPT = '#E63946'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'features_final.csv')
OUT = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(OUT, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ===========================================================
# 1. Load data, split back into train/test
# ===========================================================
df = pd.read_csv(DATA_PATH)
train_df = df[df['split'] == 'train'].drop(columns=['split'])
test_df = df[df['split'] == 'test'].drop(columns=['split'])

X_train_full = train_df.drop(columns=['class'])
y_train_full = train_df['class']
X_test = test_df.drop(columns=['class'])
y_test = test_df['class']

print(f"Train (full): {X_train_full.shape}, Test: {X_test.shape}")
print(f"Train bankruptcy rate: {y_train_full.mean()*100:.2f}%  |  Test bankruptcy rate: {y_test.mean()*100:.2f}%")

# ===========================================================
# 2. Carve a VALIDATION split out of train (leakage-safe threshold tuning)
# ===========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)
print(f"Train (fit): {X_train.shape}, Validation (threshold-tuning): {X_val.shape}, Test (final report): {X_test.shape}")

# ===========================================================
# 3. Class weight (from TRAIN-FIT only)
# ===========================================================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight (train-fit derived): {scale_pos_weight:.3f}")

# ===========================================================
# 4. Stratified 5-fold CV for both models (on train-fit portion)
# ===========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_cv_scores, cat_cv_scores = [], []

for fold, (tr_idx, cv_val_idx) in enumerate(cv.split(X_train, y_train)):
    X_tr, X_cv_val = X_train.iloc[tr_idx], X_train.iloc[cv_val_idx]
    y_tr, y_cv_val = y_train.iloc[tr_idx], y_train.iloc[cv_val_idx]

    xgb_fold = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
    xgb_fold.fit(X_tr, y_tr)
    xgb_pred = xgb_fold.predict_proba(X_cv_val)[:, 1]
    xgb_cv_scores.append(average_precision_score(y_cv_val, xgb_pred))

    cat_fold = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        class_weights=[1, scale_pos_weight], random_state=42, verbose=0
    )
    cat_fold.fit(X_tr, y_tr)
    cat_pred = cat_fold.predict_proba(X_cv_val)[:, 1]
    cat_cv_scores.append(average_precision_score(y_cv_val, cat_pred))

    print(f"Fold {fold+1}: XGBoost PR-AUC={xgb_cv_scores[-1]:.4f}  |  CatBoost PR-AUC={cat_cv_scores[-1]:.4f}")

# --- CV score graph ---
fig, ax = plt.subplots(figsize=(9.5, 6.5))
folds = np.arange(1, 6)
width = 0.35
b1 = ax.bar(folds - width/2, xgb_cv_scores, width, color=COLOR_XGB, edgecolor='black', linewidth=0.8, label='XGBoost')
b2 = ax.bar(folds + width/2, cat_cv_scores, width, color=COLOR_CAT, edgecolor='black', linewidth=0.8, label='CatBoost')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.006, f'{b.get_height():.3f}',
            ha='center', fontsize=8.5, fontweight='bold')
mean_xgb, mean_cat = np.mean(xgb_cv_scores), np.mean(cat_cv_scores)
ax.axhline(mean_xgb, color=COLOR_XGB, linestyle='--', linewidth=1.3, alpha=0.7)
ax.axhline(mean_cat, color=COLOR_CAT, linestyle='--', linewidth=1.3, alpha=0.7)
ax.set_xticks(folds)
ax.set_xlabel('CV Fold', fontsize=10.5)
ax.set_ylabel('PR-AUC (average precision)', fontsize=10.5)
ax.set_title('5-Fold Stratified Cross-Validation: XGBoost vs CatBoost', fontsize=13.5, pad=12)
legend_handles = [
    mpatches.Patch(facecolor=COLOR_XGB, edgecolor='black', label=f'XGBoost (mean={mean_xgb:.3f})'),
    mpatches.Patch(facecolor=COLOR_CAT, edgecolor='black', label=f'CatBoost (mean={mean_cat:.3f})'),
    plt.Line2D([0], [0], color='gray', linestyle='--', label='Mean line')
]
legend = ax.legend(handles=legend_handles, fontsize=9.5, frameon=True, loc='lower right')
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/M1_cv_scores.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M1_cv_scores.png")
print(f"Mean CV PR-AUC — XGBoost: {mean_xgb:.4f}  |  CatBoost: {mean_cat:.4f}")

# ===========================================================
# 5. Train final models on TRAIN-FIT portion only
# ===========================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric='logloss',
    random_state=42, n_jobs=-1
)
xgb_model.fit(X_train, y_train)

cat_model = CatBoostClassifier(
    iterations=300, depth=5, learning_rate=0.05,
    class_weights=[1, scale_pos_weight], random_state=42, verbose=0
)
cat_model.fit(X_train, y_train)

xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
cat_val_proba = cat_model.predict_proba(X_val)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]
cat_test_proba = cat_model.predict_proba(X_test)[:, 1]
print("\nBoth models trained on train-fit set (validation & test kept separate).")

# ===========================================================
# 6a. Confusion matrices on TEST at default threshold (0.5)
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, proba, name, cmap in [(axes[0], xgb_test_proba, 'XGBoost', 'Blues'),
                                (axes[1], cat_test_proba, 'CatBoost', 'Oranges')]:
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    im = ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=15, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Safe', 'Bankrupt']); ax.set_yticklabels(['Safe', 'Bankrupt'])
    ax.set_xlabel('Predicted', fontsize=10.5); ax.set_ylabel('Actual', fontsize=10.5)
    ax.set_title(f'{name} (threshold=0.5)', fontsize=12.5,
                 color=COLOR_XGB if name == 'XGBoost' else COLOR_CAT)
fig.suptitle('Confusion Matrices on TEST Set at Default Threshold (0.5)', fontsize=14.5, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/M2_confusion_matrices.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M2_confusion_matrices.png")

# ===========================================================
# 6b. ROC curve (TEST set)
# ===========================================================
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for proba, name, color in [(xgb_test_proba, 'XGBoost', COLOR_XGB), (cat_test_proba, 'CatBoost', COLOR_CAT)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.4, label=f'{name} (AUC={auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.2, label='Random guess (AUC=0.500)')
ax.fill_between([0, 1], [0, 1], alpha=0.03, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=10.5)
ax.set_ylabel('True Positive Rate', fontsize=10.5)
ax.set_title('ROC Curve — XGBoost vs CatBoost (Test Set)', fontsize=13.5, pad=12)
legend = ax.legend(fontsize=10, loc='lower right', frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/M3_roc_curve.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M3_roc_curve.png")

# ===========================================================
# 6c. Precision-Recall curve (TEST set)
# ===========================================================
fig, ax = plt.subplots(figsize=(8.5, 7.5))
for proba, name, color in [(xgb_test_proba, 'XGBoost', COLOR_XGB), (cat_test_proba, 'CatBoost', COLOR_CAT)]:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ap = average_precision_score(y_test, proba)
    ax.plot(recall, precision, color=color, linewidth=2.4, label=f'{name} (AP={ap:.3f})')
baseline = y_test.mean()
ax.axhline(baseline, color='gray', linestyle='--', linewidth=1.2, label=f'Baseline (positive rate={baseline:.3f})')
ax.set_xlabel('Recall', fontsize=10.5)
ax.set_ylabel('Precision', fontsize=10.5)
ax.set_title('Precision-Recall Curve — XGBoost vs CatBoost (Test Set)', fontsize=13.5, pad=12)
legend = ax.legend(fontsize=10, loc='upper right', frameon=True)
legend.get_frame().set_edgecolor('black')
plt.tight_layout()
plt.savefig(f'{OUT}/M4_pr_curve.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M4_pr_curve.png")

# ===========================================================
# 7. Threshold tuning on VALIDATION set ONLY (leakage-safe)
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5))
best_thresholds = {}

for ax, (val_proba, name, color) in zip(
        axes, [(xgb_val_proba, 'XGBoost', COLOR_XGB), (cat_val_proba, 'CatBoost', COLOR_CAT)]):
    precision, recall, thresholds = precision_recall_curve(y_val, val_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])
    best_thresh = thresholds[best_idx]
    best_thresholds[name] = best_thresh

    ax.plot(thresholds, precision[:-1], color='#264653', linewidth=1.8, label='Precision')
    ax.plot(thresholds, recall[:-1], color='#2A9D8F', linewidth=1.8, label='Recall')
    ax.plot(thresholds, f1_scores[:-1], color=color, linewidth=2.6, label='F1-score')
    ax.axvline(best_thresh, color='black', linestyle='--', linewidth=1.3)
    ax.scatter([best_thresh], [f1_scores[best_idx]], color=color, s=100, zorder=5, edgecolors='black', linewidths=1.2)
    ax.annotate(f'Best threshold = {best_thresh:.3f}\nF1 (on validation) = {f1_scores[best_idx]:.3f}',
                xy=(best_thresh, f1_scores[best_idx]), xytext=(0.42, 0.15), textcoords='axes fraction',
                fontsize=9.5, fontweight='bold', arrowprops=dict(arrowstyle='->', lw=1.3, color=color),
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=color))
    ax.set_xlabel('Threshold', fontsize=10.5)
    ax.set_ylabel('Score', fontsize=10.5)
    ax.set_title(f'{name}: Threshold Tuning (on VALIDATION set)', fontsize=12, color=color)
    legend = ax.legend(fontsize=9.5, loc='center left', frameon=True)
    legend.get_frame().set_edgecolor('black')

fig.suptitle('Finding the Optimal Decision Threshold — Tuned on Validation, Never on Test',
             fontsize=14.5, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/M5_threshold_tuning.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M5_threshold_tuning.png")
print(f"Best thresholds (chosen on validation) — XGBoost: {best_thresholds['XGBoost']:.3f}  |  "
      f"CatBoost: {best_thresholds['CatBoost']:.3f}")

# ===========================================================
# 8. Confusion matrices on TEST at TUNED threshold
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
for ax, proba, name, cmap in [(axes[0], xgb_test_proba, 'XGBoost', 'Blues'),
                                (axes[1], cat_test_proba, 'CatBoost', 'Oranges')]:
    thresh = best_thresholds[name]
    preds = (proba >= thresh).astype(int)
    cm = confusion_matrix(y_test, preds)
    im = ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', fontsize=15, fontweight='bold',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Safe', 'Bankrupt']); ax.set_yticklabels(['Safe', 'Bankrupt'])
    ax.set_xlabel('Predicted', fontsize=10.5); ax.set_ylabel('Actual', fontsize=10.5)
    ax.set_title(f'{name} (threshold={thresh:.3f}, from validation)', fontsize=11.5,
                 color=COLOR_XGB if name == 'XGBoost' else COLOR_CAT)
fig.suptitle('Confusion Matrices on TEST Set at Tuned Threshold (leakage-safe)',
             fontsize=14.5, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT}/M6_confusion_matrices_tuned.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M6_confusion_matrices_tuned.png")

# ===========================================================
# 9. Feature importance (both models)
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.5))
xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
cat_imp = pd.Series(cat_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)

for ax, imp, name, color in [(axes[0], xgb_imp, 'XGBoost', COLOR_XGB), (axes[1], cat_imp, 'CatBoost', COLOR_CAT)]:
    labels = [c if len(c) <= 28 else c[:25] + '...' for c in imp.index[::-1]]
    ax.barh(range(len(imp)), imp.values[::-1], color=color, edgecolor='black', linewidth=0.6, alpha=0.88)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Importance', fontsize=10.5)
    ax.set_title(f'{name}: Top 15 Feature Importances', fontsize=12.5, color=color)
    top_feat = imp.index[0]
    ax.annotate(f'Most important:\n{top_feat[:30]}',
                xy=(imp.values[0], len(imp)-1), xytext=(0.4, 0.15), textcoords='axes fraction',
                fontsize=8.5, fontweight='bold', color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color))

fig.suptitle('Feature Importance Comparison: XGBoost vs CatBoost', fontsize=14.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/M7_feature_importance.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M7_feature_importance.png")

# ===========================================================
# 10. Final comparison table (test set only, threshold chosen on validation)
# ===========================================================
results = []
for test_proba, name in [(xgb_test_proba, 'XGBoost'), (cat_test_proba, 'CatBoost')]:
    for label, thresh in [('default (0.5)', 0.5), ('tuned (validation)', float(best_thresholds[name]))]:
        preds = (test_proba >= thresh).astype(int)
        results.append({
            'model': name, 'threshold_type': label, 'threshold': round(float(thresh), 3),
            'accuracy': round(accuracy_score(y_test, preds), 4),
            'precision': round(precision_score(y_test, preds), 4),
            'recall': round(recall_score(y_test, preds), 4),
            'f1': round(f1_score(y_test, preds), 4),
            'roc_auc': round(roc_auc_score(y_test, test_proba), 4),
            'pr_auc': round(average_precision_score(y_test, test_proba), 4),
        })

comparison_df = pd.DataFrame(results)
print("\n=== Final Comparison Table (test set, threshold chosen on validation - no leakage) ===")
print(comparison_df)

# --- Comparison table as a clean image ---
fig, ax = plt.subplots(figsize=(13, 3.2))
ax.axis('off')
tbl_cols = ['model', 'threshold_type', 'threshold', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
cell_text = comparison_df[tbl_cols].values
row_colors = []
for r in comparison_df['model']:
    row_colors.append('#EAF3F8' if r == 'XGBoost' else '#FDEEE9')
table = ax.table(cellText=cell_text, colLabels=tbl_cols, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.1)
for j, colname in enumerate(tbl_cols):
    width = 0.19 if colname == 'threshold_type' else 0.10
    for i in range(len(cell_text) + 1):
        table[i, j].set_width(width)
for i in range(len(cell_text)):
    for j in range(len(tbl_cols)):
        table[i+1, j].set_facecolor(row_colors[i])
        table[i+1, j].set_edgecolor('black')
for j in range(len(tbl_cols)):
    table[0, j].set_facecolor('#264653')
    table[0, j].set_text_props(color='white', fontweight='bold')
    table[0, j].set_edgecolor('black')
ax.set_title('Final Model Comparison — Test Set (threshold tuned on validation only)',
             fontsize=13, fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig(f'{OUT}/M8_comparison_table.png', dpi=170, bbox_inches='tight')
plt.close()
print("Saved: M8_comparison_table.png")

REPORTS_OUT = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_OUT, exist_ok=True)
comparison_df.to_csv(f'{REPORTS_OUT}/model_comparison.csv', index=False)
print(f"\nSaved comparison table: {REPORTS_OUT}/model_comparison.csv")

print("\n=== Model training complete: 8 graphs + 1 comparison CSV (leakage-safe threshold tuning) ===")


# 11. Save trained models (.pkl) and per-company predictions (.csv)
# ===========================================================
# Why: without this, every time you want to reuse these exact models
# (e.g. to compare against the Optuna-tuned version later, or to run
# the explainability/SHAP step) you'd have to retrain from scratch.
# Saving predictions per-company also lets you inspect individual
# cases later without re-running the whole pipeline.
 
import joblib
 
MODELS_OUT = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_OUT, exist_ok=True)
 
# --- Save the trained model objects ---
joblib.dump(xgb_model, f'{MODELS_OUT}/xgboost_baseline.pkl')
joblib.dump(cat_model, f'{MODELS_OUT}/catboost_baseline.pkl')
print(f"\nSaved models to: {MODELS_OUT}/xgboost_baseline.pkl, catboost_baseline.pkl")
 
# --- Save per-company predictions on the TEST set ---
predictions_df = pd.DataFrame({
    'actual_class': y_test.values,
    'xgb_probability': xgb_test_proba,
    'xgb_predicted_default': (xgb_test_proba >= 0.5).astype(int),
    'xgb_predicted_tuned': (xgb_test_proba >= best_thresholds['XGBoost']).astype(int),
    'cat_probability': cat_test_proba,
    'cat_predicted_default': (cat_test_proba >= 0.5).astype(int),
    'cat_predicted_tuned': (cat_test_proba >= best_thresholds['CatBoost']).astype(int),
}, index=X_test.index)
 
predictions_df.to_csv(f'{REPORTS_OUT}/test_predictions.csv', index=True)
print(f"Saved per-company predictions to: {REPORTS_OUT}/test_predictions.csv")
 
# --- Also save the CV fold scores (raw numbers, not just the graph) ---
cv_scores_df = pd.DataFrame({
    'fold': range(1, 6),
    'xgboost_pr_auc': xgb_cv_scores,
    'catboost_pr_auc': cat_cv_scores
})
cv_scores_df.to_csv(f'{REPORTS_OUT}/cv_scores.csv', index=False)
print(f"Saved CV fold scores to: {REPORTS_OUT}/cv_scores.csv")
 
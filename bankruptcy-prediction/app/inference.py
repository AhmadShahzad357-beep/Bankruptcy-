"""
app/inference.py

Core prediction logic, shared by both the dashboard and the API.

Given a raw company's financial ratios (the same 65 ratios + forecasting
period from the original dataset), this module:
  1. Applies the exact same preprocessing the model was trained on
     (missing indicators, imputation, outlier capping, VIF-based
     column selection, Altman Z-Score, category aggregates, one-hot
     encoding of forecasting period) - using the saved artifacts so
     nothing drifts from what happened during training.
  2. Runs the trained XGBoost model to get a bankruptcy probability.
  3. Uses SHAP to explain the prediction (top contributing factors).
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'xgboost_baseline.pkl')
ARTIFACTS_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'preprocessing_artifacts.pkl')

THRESHOLD = 0.679  # validation-tuned threshold for xgboost_baseline.pkl (see reports/model_comparison.csv)

_model = None
_artifacts = None
_explainer = None


def _load():
    """Load model, artifacts, and SHAP explainer once (cached)."""
    global _model, _artifacts, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _artifacts = joblib.load(ARTIFACTS_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _artifacts, _explainer


def get_raw_input_template():
    """Returns the list of raw fields a caller needs to supply -
    useful for building a form or API schema."""
    _, artifacts, _ = _load()
    return artifacts['ratio_cols_original'] + ['forecasting period']


def _transform_raw_to_features(raw_row: dict) -> pd.DataFrame:
    """Apply the exact training-time preprocessing to one raw company record.

    raw_row: dict with the 65 raw ratio names + 'forecasting period' as keys.
    Missing keys are treated as missing values (will be imputed).
    """
    _, artifacts, _ = _load()

    ratio_cols_original = artifacts['ratio_cols_original']
    row = pd.DataFrame([raw_row])

    # Ensure every expected raw column exists (missing -> NaN)
    for col in ratio_cols_original + ['forecasting period']:
        if col not in row.columns:
            row[col] = np.nan
    row = row[ratio_cols_original + ['forecasting period']].apply(pd.to_numeric, errors='coerce')

    # 1. Missing-indicator flags (only for columns that had them at train time)
    for col in artifacts['indicator_cols']:
        row[f'{col}_was_missing'] = row[col].isna().astype(int)

    # 2. Median imputation (train-derived medians)
    for col in ratio_cols_original:
        if row[col].isna().any():
            row[col] = row[col].fillna(artifacts['medians'][col])

    # 3. Outlier capping (train-derived 1st/99th percentile bounds)
    for col in ratio_cols_original:
        lo, hi = artifacts['lower_bounds'][col], artifacts['upper_bounds'][col]
        row[col] = row[col].clip(lo, hi)

    # 4. Altman Z-Score (computed before VIF drop, so its ingredients are safe)
    row['altman_z'] = (
        1.2 * row['working_capital_over_total_assets'] +
        1.4 * row['retained_earnings_over_total_assets'] +
        3.3 * row['EBIT_over_total_assets'] +
        0.6 * row['book_value_of_equity_over_total_liabilities'] +
        1.0 * row['sales_over_total_assets']
    )

    # 5. Keep only VIF-surviving ratio columns
    kept_ratio_cols = artifacts['kept_ratio_cols']

    # 6. Category-wise aggregate features
    cat_map = artifacts['cat_map']
    for cat in sorted(set(cat_map.values())):
        cols_in_cat = [c for c, v in cat_map.items() if v == cat and c in kept_ratio_cols]
        row[f'agg_{cat}_mean'] = row[cols_in_cat].mean(axis=1)

    # 7. One-hot encode forecasting period (matching train-time categories)
    for p in artifacts['period_values']:
        row[f'period_{int(p)}'] = (row['forecasting period'] == p).astype(int)

    # 8. Assemble final feature vector in the EXACT column order the model expects
    final_cols = artifacts['final_feature_columns']
    row = row.reindex(columns=final_cols, fill_value=0)

    return row


def predict_company(raw_row: dict, top_n_factors: int = 5) -> dict:
    """Main entry point: raw company ratios in, full prediction + explanation out."""
    model, artifacts, explainer = _load()

    features = _transform_raw_to_features(raw_row)
    proba = float(model.predict_proba(features)[0, 1])
    predicted_bankrupt = bool(proba >= THRESHOLD)

    if proba < 0.3:
        risk_level = 'Low'
    elif proba < THRESHOLD:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    shap_values = explainer.shap_values(features)[0]
    feature_names = features.columns.tolist()
    contributions = pd.Series(shap_values, index=feature_names).sort_values(key=abs, ascending=False)

    top_factors = []
    for feat, val in contributions.head(top_n_factors).items():
        top_factors.append({
            'feature': feat,
            'shap_value': round(float(val), 4),
            'direction': 'increases risk' if val > 0 else 'decreases risk',
            'raw_value': round(float(features.iloc[0][feat]), 4),
        })

    return {
        'bankruptcy_probability': round(proba, 4),
        'predicted_bankrupt': predicted_bankrupt,
        'risk_level': risk_level,
        'threshold_used': THRESHOLD,
        'top_factors': top_factors,
    }


if __name__ == '__main__':
    # Quick self-test using a real row from the training data
    import sys
    test_csv = os.path.join(PROJECT_ROOT, 'data', 'raw', 'train.csv')
    sample = pd.read_csv(test_csv, low_memory=False).iloc[0].to_dict()
    sample.pop('id', None)
    sample.pop('class', None)
    result = predict_company(sample)
    print("Self-test prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
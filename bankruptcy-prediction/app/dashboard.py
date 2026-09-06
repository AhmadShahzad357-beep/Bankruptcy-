"""
app/dashboard.py

Interactive web dashboard for the bankruptcy risk model.
Run with:  streamlit run app/dashboard.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import predict_company, get_raw_input_template, _load

# ===========================================================
# Page config + custom CSS (professional look)
# ===========================================================
st.set_page_config(page_title="Bankruptcy Risk Dashboard", page_icon="📊", layout="wide")

COLOR_SAFE = "#2A9D8F"
COLOR_MEDIUM = "#E9C46A"
COLOR_HIGH = "#E63946"
COLOR_PRIMARY = "#264653"
COLOR_ACCENT = "#2E86AB"

st.markdown(f"""
<style>
    .main {{ background-color: #F7F9FA; }}
    .block-container {{ padding-top: 2rem; }}

    .dashboard-header {{
        background: linear-gradient(135deg, {COLOR_PRIMARY} 0%, {COLOR_ACCENT} 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }}
    .dashboard-header h1 {{
        margin: 0;
        font-size: 1.9rem;
        font-weight: 700;
    }}
    .dashboard-header p {{
        margin: 0.4rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }}

    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 5px solid {COLOR_ACCENT};
        margin-bottom: 1rem;
    }}
    .metric-card h3 {{
        margin: 0 0 0.3rem 0;
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .metric-card .value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
    }}

    .risk-badge {{
        display: inline-block;
        padding: 0.5rem 1.4rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
    }}

    .factor-row {{
        display: flex;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid #eee;
    }}
    .factor-name {{
        flex: 1;
        font-size: 0.88rem;
        color: #333;
    }}
    .factor-tag {{
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        color: white;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {COLOR_PRIMARY};
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
</style>
""", unsafe_allow_html=True)

# ===========================================================
# Header
# ===========================================================
st.markdown("""
<div class="dashboard-header">
    <h1>📊 Corporate Bankruptcy Risk Dashboard</h1>
    <p>Explainable early-warning system — enter a company's financials or upload a CSV to assess bankruptcy risk.</p>
</div>
""", unsafe_allow_html=True)

# ===========================================================
# Load model/artifacts once, get medians for pre-filling the form
# ===========================================================
_, artifacts, _ = _load()
medians = artifacts["medians"]

# Curated set of the most informative raw ratios to show in the manual form
# (rest of the 65 raw fields are auto-filled with training medians if left untouched)
KEY_FIELDS = [
    "net_profit_over_total_assets",
    "total_liabilities_over_total_assets",
    "working_capital_over_total_assets",
    "retained_earnings_over_total_assets",
    "EBIT_over_total_assets",
    "sales_over_total_assets",
    "profit_on_sales_over_sales",
    "current_assets_over_short_term_liabilities",
    "equity_over_total_assets",
    "short_term_liabilities_over_total_assets",
    "profit_on_operating_activities_over_financial_expenses",
    "operating_expenses_over_total_liabilities",
]

# ===========================================================
# Sidebar - input mode
# ===========================================================
st.sidebar.markdown("### ⚙️ Input Mode")
mode = st.sidebar.radio("Choose how to provide company data:", ["Manual entry (key ratios)", "Upload CSV"])
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**About this tool**\n\n"
    "Trained on 25,121 companies (Taiwan Economic Journal dataset). "
    "Predictions use a tuned XGBoost model; explanations use SHAP.\n\n"
    "Fields left at their default (median) value are treated as typical, not missing."
)

# ===========================================================
# Helper: render one prediction result nicely
# ===========================================================
def render_result(result: dict):
    prob = result["bankruptcy_probability"]
    level = result["risk_level"]
    level_color = {"Low": COLOR_SAFE, "Medium": COLOR_MEDIUM, "High": COLOR_HIGH}[level]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Bankruptcy Probability</h3>
            <div class="value">{prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Risk Level</h3>
            <span class="risk-badge" style="background-color:{level_color};">{level}</span>
        </div>""", unsafe_allow_html=True)
    with col3:
        verdict = "Bankrupt (flagged)" if result["predicted_bankrupt"] else "Safe (not flagged)"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Verdict (threshold={result['threshold_used']})</h3>
            <div class="value" style="font-size:1.3rem;">{verdict}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("#### Why this prediction? (Top contributing factors)")

    factors = result["top_factors"]
    names = [f["feature"] if len(f["feature"]) <= 42 else f["feature"][:39] + "..." for f in factors]
    vals = [f["shap_value"] for f in factors]
    colors = [COLOR_HIGH if v > 0 else COLOR_SAFE for v in vals]

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(factors) + 1))
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    y_pos = np.arange(len(factors))
    ax.barh(y_pos, vals[::-1], color=colors[::-1], edgecolor='black', linewidth=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('SHAP value (impact on prediction)', fontsize=9.5)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:+.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=8, fontweight='bold')
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_HIGH, edgecolor='black', label='Increases risk'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_SAFE, edgecolor='black', label='Decreases risk'),
    ]
    ax.legend(handles=legend_handles, fontsize=8.5, loc='lower right', frameon=True)
    fig.tight_layout()
    st.pyplot(fig)


# ===========================================================
# Manual entry mode
# ===========================================================
if mode == "Manual entry (key ratios)":
    st.markdown("### ✍️ Enter Key Financial Ratios")
    st.caption("Fields are pre-filled with the training-set median. Adjust the ones you know; leave the rest as-is.")

    cols = st.columns(3)
    values = {}
    for i, field in enumerate(KEY_FIELDS):
        with cols[i % 3]:
            default = float(medians.get(field, 0.0))
            values[field] = st.number_input(field.replace("_", " "), value=round(default, 4), format="%.4f")

    period = st.selectbox("Forecasting period", [1, 2, 3, 4], index=1)
    values["forecasting period"] = period

    if st.button("🔍 Assess Risk", type="primary"):
        result = predict_company(values)
        st.markdown("---")
        render_result(result)

# ===========================================================
# CSV upload mode
# ===========================================================
else:
    st.markdown("### 📁 Upload Company Data (CSV)")
    st.caption("CSV should contain the raw financial ratio columns (same schema as the training data). "
               "Missing columns are automatically imputed.")

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(input_df)} row(s).")
        st.dataframe(input_df.head())

        if st.button("🔍 Assess Risk for All Rows", type="primary"):
            results = []
            for _, row in input_df.iterrows():
                r = predict_company(row.to_dict())
                results.append({
                    "bankruptcy_probability": r["bankruptcy_probability"],
                    "risk_level": r["risk_level"],
                    "predicted_bankrupt": r["predicted_bankrupt"],
                })
            results_df = pd.DataFrame(results)
            combined = pd.concat([input_df.reset_index(drop=True), results_df], axis=1)

            st.markdown("#### Results")
            st.dataframe(
                combined.style.applymap(
                    lambda v: f"background-color: {COLOR_HIGH}33" if v == "High" else
                              (f"background-color: {COLOR_MEDIUM}33" if v == "Medium" else
                               f"background-color: {COLOR_SAFE}33"),
                    subset=["risk_level"]
                )
            )

            if len(input_df) == 1:
                st.markdown("---")
                render_result(predict_company(input_df.iloc[0].to_dict()))
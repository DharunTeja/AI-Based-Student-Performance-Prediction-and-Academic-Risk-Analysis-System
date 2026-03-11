"""
5_📜_History.py - Prediction History Page.

This page displays:
1. All past prediction records
2. Summary statistics of predictions
3. Risk level trends over time
4. Ability to view detailed individual predictions
5. Option to clear history
6. Export history as CSV

Data is retrieved from Supabase or local JSON storage.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.db_manager import get_prediction_history, clear_prediction_history

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="History | Student Performance",
    page_icon="📜",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0F0C29 0%, #302B63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 10px 0;
    }
    .page-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C896 0%, #00B4D8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #8888a0;
        margin-bottom: 25px;
    }
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,200,150,0.3), transparent);
        margin: 25px 0;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
    }
    [data-testid="stMetric"] label { color: #a0a0b0 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
    .stButton > button {
        background: linear-gradient(135deg, #00C896 0%, #00B4D8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .history-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 15px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .history-row:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0,200,150,0.2);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-title">📜 Prediction History</div>
<div class="page-subtitle">View and analyze all past student performance predictions</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD PREDICTION HISTORY
# ============================================================
history = get_prediction_history(limit=100)

if not history or len(history) == 0:
    # No predictions yet
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 15px;">📭</div>
        <h3 style="color: #FFC857;">No Predictions Yet</h3>
        <p style="color: #a0a0b0;">
            Start making predictions on the <strong style="color: #00C896;">Predict</strong> 
            or <strong style="color: #00B4D8;">Batch Upload</strong> page to see history here.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# SUMMARY METRICS
# ============================================================
st.markdown("### 📊 History Summary")

# Convert history to DataFrame for analysis
history_df = pd.DataFrame(history)

total = len(history_df)
pass_count = (history_df["prediction"] == "Pass").sum()
fail_count = (history_df["prediction"] == "Fail").sum()
high_risk = (history_df["risk_level"] == "High Risk").sum()
medium_risk = (history_df["risk_level"] == "Medium Risk").sum()
low_risk = (history_df["risk_level"] == "Low Risk").sum()

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("Total Predictions", f"{total}")
with m2:
    st.metric("Pass", f"{pass_count}", delta=f"{pass_count/total*100:.0f}%")
with m3:
    st.metric("Fail", f"{fail_count}", delta=f"-{fail_count/total*100:.0f}%")
with m4:
    st.metric("🔴 High Risk", f"{high_risk}")
with m5:
    st.metric("🟡 Medium Risk", f"{medium_risk}")
with m6:
    st.metric("🟢 Low Risk", f"{low_risk}")

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ============================================================
# CHARTS
# ============================================================
col1, col2 = st.columns(2)

with col1:
    # Prediction distribution pie chart
    pred_counts = history_df["prediction"].value_counts()
    fig_pred = px.pie(
        values=pred_counts.values,
        names=pred_counts.index,
        title="🎯 Prediction Distribution",
        color=pred_counts.index,
        color_discrete_map={"Pass": "#00C896", "Fail": "#FF4B4B"},
        hole=0.45,
    )
    fig_pred.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0a0b0"},
    )
    fig_pred.update_traces(textinfo="label+percent+value")
    st.plotly_chart(fig_pred, use_container_width=True)

with col2:
    # Risk level distribution
    risk_counts = history_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig_risk = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        title="⚠️ Risk Level Distribution",
        color="Risk Level",
        color_discrete_map=config.RISK_COLORS,
        text="Count",
    )
    fig_risk.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0a0b0"},
    )
    fig_risk.update_traces(textposition="outside", textfont_size=14)
    st.plotly_chart(fig_risk, use_container_width=True)

# Probability distribution histogram
if "pass_probability" in history_df.columns:
    fig_prob = px.histogram(
        history_df,
        x="pass_probability",
        nbins=20,
        title="📊 Pass Probability Distribution Across All Predictions",
        color_discrete_sequence=["#00B4D8"],
        opacity=0.8,
    )
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a0a0b0"},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)", "title": "Pass Probability"},
        yaxis={"gridcolor": "rgba(255,255,255,0.05)", "title": "Count"},
    )
    fig_prob.add_vline(x=0.4, line_dash="dash", line_color="#FF4B4B",
                       annotation_text="High Risk", annotation_font_color="#FF4B4B")
    fig_prob.add_vline(x=0.7, line_dash="dash", line_color="#FFC857",
                       annotation_text="Low Risk", annotation_font_color="#FFC857")
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ============================================================
# HISTORY TABLE
# ============================================================
st.markdown("### 📋 Prediction Records")

# Filter options in sidebar
with st.sidebar:
    st.markdown("### 🔍 Filters")
    
    filter_prediction = st.multiselect(
        "Prediction Result",
        options=["Pass", "Fail"],
        default=["Pass", "Fail"],
    )
    
    filter_risk = st.multiselect(
        "Risk Level",
        options=["High Risk", "Medium Risk", "Low Risk"],
        default=["High Risk", "Medium Risk", "Low Risk"],
    )

# Apply filters
filtered_df = history_df[
    (history_df["prediction"].isin(filter_prediction)) &
    (history_df["risk_level"].isin(filter_risk))
]

st.markdown(f"""
<div style="color: #a0a0b0; font-size: 0.85rem; margin-bottom: 10px;">
    Showing <strong style="color: #00C896;">{len(filtered_df)}</strong> of 
    <strong style="color: #00B4D8;">{total}</strong> records
</div>
""", unsafe_allow_html=True)

# Display columns to show
display_cols = ["timestamp", "prediction", "risk_level", "pass_probability", "recommendations_count"]
available_cols = [col for col in display_cols if col in filtered_df.columns]

if available_cols:
    display_df = filtered_df[available_cols].copy()
    
    # Rename columns for display
    display_df.columns = [
        col.replace("_", " ").title() for col in display_df.columns
    ]
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
    )

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ============================================================
# ACTIONS
# ============================================================
st.markdown("### ⚡ Actions")

col_actions1, col_actions2 = st.columns(2)

with col_actions1:
    # Download history as CSV
    if len(filtered_df) > 0:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

with col_actions2:
    # Clear history button (with confirmation)
    if st.button("🗑️ Clear All History", use_container_width=True):
        # Use a session state flag for confirmation
        st.session_state["confirm_clear"] = True

# Confirmation dialog
if st.session_state.get("confirm_clear", False):
    st.warning("⚠️ Are you sure you want to clear ALL prediction history?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Yes, Clear", use_container_width=True):
            if clear_prediction_history():
                st.success("✅ History cleared successfully!")
                st.session_state["confirm_clear"] = False
                st.rerun()  # Refresh the page
    with c2:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state["confirm_clear"] = False
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-divider"></div>
<div style="text-align: center; color: #555580; font-size: 0.75rem; padding-bottom: 20px;">
    Prediction History — AI Student Performance Prediction System
</div>
""", unsafe_allow_html=True)

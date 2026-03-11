"""
1_📊_Dashboard.py - Analytics Dashboard Page.

This page provides interactive data visualizations and analytics:
1. Dataset overview with key statistics
2. Grade distribution charts
3. Pass/Fail ratio visualization
4. Feature correlation heatmap
5. Attendance vs Performance analysis
6. Risk level distribution
7. Comparative analysis across datasets

Uses Plotly for interactive, beautiful charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px           # High-level plotting library
import plotly.graph_objects as go     # Low-level for custom charts
from plotly.subplots import make_subplots  # For multi-chart layouts
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_preprocessing import (
    load_primary_dataset,
    load_secondary_dataset_1,
    load_secondary_dataset_2,
    get_risk_level,
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Dashboard | Student Performance",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (Same theme as main app)
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
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 200, 150, 0.15);
    }
    [data-testid="stMetric"] label { color: #a0a0b0 !important; }
    [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 700; }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-title">📊 Analytics Dashboard</div>
<div class="page-subtitle">Explore interactive visualizations of student academic data</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# PLOTLY CHART THEME
# ============================================================
# Define a consistent dark theme for all Plotly charts
CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",       # Transparent background
    "plot_bgcolor": "rgba(0,0,0,0)",         # Transparent plot area
    "font": {"color": "#a0a0b0", "family": "Inter"},  # Light text
    "title_font": {"color": "#ffffff", "size": 16},    # White titles
    "xaxis": {
        "gridcolor": "rgba(255,255,255,0.05)",  # Subtle grid lines
        "zerolinecolor": "rgba(255,255,255,0.1)",
    },
    "yaxis": {
        "gridcolor": "rgba(255,255,255,0.05)",
        "zerolinecolor": "rgba(255,255,255,0.1)",
    },
}

# Custom color palette for charts (vibrant, modern colors)
COLOR_PALETTE = ["#00C896", "#00B4D8", "#FF6B6B", "#FFC857", "#845EC2", "#FF9671", "#D65DB1"]


# ============================================================
# LOAD DATASETS
# ============================================================
@st.cache_data  # Cache the data so it doesn't reload on every interaction
def load_all_datasets():
    """Load and cache all datasets for the dashboard."""
    primary = load_primary_dataset()
    secondary1 = load_secondary_dataset_1()
    secondary2 = load_secondary_dataset_2()
    return primary, secondary1, secondary2


# Load datasets
df_primary, df_secondary1, df_secondary2 = load_all_datasets()

# ============================================================
# DATASET SELECTOR (in sidebar)
# ============================================================
with st.sidebar:
    st.markdown("### 📂 Dataset Selector")
    dataset_choice = st.selectbox(
        "Select dataset to visualize",
        ["Primary (student-mat.csv)", "Performance Index", "Exam Scores"],
        help="Choose which dataset to analyze in the dashboard"
    )

# ============================================================
# PRIMARY DATASET ANALYTICS
# ============================================================
if dataset_choice == "Primary (student-mat.csv)" and df_primary is not None:
    df = df_primary.copy()
    
    # Add derived columns for analysis
    df["pass_fail"] = df["G3"].apply(lambda x: "Pass" if x >= config.PASS_THRESHOLD else "Fail")
    df["risk_level"] = df["G3"].apply(
        lambda x: get_risk_level(x / 20.0)  # Convert grade to probability scale
    )
    
    # --- OVERVIEW METRICS ---
    st.markdown("### 📈 Dataset Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    
    with m1:
        st.metric("Total Students", f"{len(df):,}")
    with m2:
        st.metric("Average Grade (G3)", f"{df['G3'].mean():.1f}/20")
    with m3:
        pass_rate = (df['pass_fail'] == 'Pass').mean() * 100
        st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with m4:
        st.metric("Avg Absences", f"{df['absences'].mean():.1f}")
    with m5:
        st.metric("Features", f"{len(df.columns)}")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # --- ROW 1: Grade Distribution + Pass/Fail Pie Chart ---
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Histogram of final grades (G3) with pass/fail coloring
        fig_grade = px.histogram(
            df, x="G3",
            color="pass_fail",                    # Color by Pass/Fail
            nbins=20,                              # 20 bins for 0-20 scale
            title="📊 Final Grade (G3) Distribution",
            labels={"G3": "Final Grade (0-20)", "pass_fail": "Result"},
            color_discrete_map={"Pass": "#00C896", "Fail": "#FF4B4B"},
            barmode="overlay",                     # Overlapping bars
            opacity=0.8,
        )
        fig_grade.update_layout(**CHART_THEME)
        # Add a vertical line at the pass threshold
        fig_grade.add_vline(
            x=config.PASS_THRESHOLD,              # Line at grade 10
            line_dash="dash",                      # Dashed line
            line_color="#FFC857",                   # Yellow color
            annotation_text="Pass Threshold",      # Label
            annotation_font_color="#FFC857",
        )
        st.plotly_chart(fig_grade, use_container_width=True)
    
    with col2:
        # Pie chart showing Pass vs Fail distribution
        pass_fail_counts = df["pass_fail"].value_counts()
        fig_pie = px.pie(
            values=pass_fail_counts.values,
            names=pass_fail_counts.index,
            title="🎯 Pass/Fail Distribution",
            color=pass_fail_counts.index,
            color_discrete_map={"Pass": "#00C896", "Fail": "#FF4B4B"},
            hole=0.4,  # Donut chart (hole in the middle)
        )
        fig_pie.update_layout(**CHART_THEME)
        fig_pie.update_traces(
            textinfo="label+percent",  # Show label and percentage
            textfont_size=14,
            marker=dict(line=dict(color='rgba(0,0,0,0.3)', width=2))
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # --- ROW 2: Risk Level Distribution + Grade Trend ---
    col3, col4 = st.columns(2)
    
    with col3:
        # Bar chart showing students in each risk category
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            title="⚠️ Risk Level Distribution",
            color="Risk Level",
            color_discrete_map=config.RISK_COLORS,  # Red, Orange, Green
            text="Count",  # Show count on bars
        )
        fig_risk.update_layout(**CHART_THEME)
        fig_risk.update_traces(textposition="outside", textfont_size=14)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col4:
        # Box plot showing grade progression across G1, G2, G3
        # This shows how grades change from first to last period
        grade_data = df[["G1", "G2", "G3"]].melt(
            var_name="Period",        # Column name for G1/G2/G3
            value_name="Grade",       # Column name for the grade values
        )
        fig_trend = px.box(
            grade_data,
            x="Period",
            y="Grade",
            title="📈 Grade Progression (G1 → G2 → G3)",
            color="Period",
            color_discrete_sequence=COLOR_PALETTE[:3],
        )
        fig_trend.update_layout(**CHART_THEME)
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # --- ROW 3: Absences vs Grade + Study Time Impact ---
    col5, col6 = st.columns(2)
    
    with col5:
        # Scatter plot: Absences vs Final Grade (G3)
        # Shows the relationship between missing classes and performance
        fig_abs = px.scatter(
            df, x="absences", y="G3",
            color="pass_fail",
            title="📉 Absences vs Final Grade",
            labels={"absences": "Number of Absences", "G3": "Final Grade (G3)"},
            color_discrete_map={"Pass": "#00C896", "Fail": "#FF4B4B"},
            opacity=0.6,
        )
        # Add a manual trend line using numpy (no statsmodels dependency)
        # np.polyfit computes a linear best-fit: y = mx + b
        z = np.polyfit(df["absences"], df["G3"], 1)  # degree=1 for linear
        x_line = np.linspace(df["absences"].min(), df["absences"].max(), 100)
        y_line = np.polyval(z, x_line)  # Evaluate the polynomial at x_line points
        fig_abs.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color="#FFC857", width=2, dash="dash"),  # Yellow dashed line
        ))
        fig_abs.update_layout(**CHART_THEME)
        st.plotly_chart(fig_abs, use_container_width=True)
    
    with col6:
        # Bar chart: Average grade by study time level
        # studytime: 1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs
        study_labels = {1: "< 2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: "> 10 hrs"}
        study_avg = df.groupby("studytime")["G3"].mean().reset_index()
        study_avg["Study Time"] = study_avg["studytime"].map(study_labels)
        
        fig_study = px.bar(
            study_avg,
            x="Study Time",
            y="G3",
            title="⏰ Average Grade by Study Time",
            labels={"G3": "Average Final Grade"},
            color="G3",
            color_continuous_scale=["#FF4B4B", "#FFC857", "#00C896"],
            text=study_avg["G3"].round(1),
        )
        fig_study.update_layout(**CHART_THEME)
        fig_study.update_traces(textposition="outside", textfont_size=13)
        st.plotly_chart(fig_study, use_container_width=True)
    
    # --- ROW 4: Feature Correlations ---
    st.markdown("### 🔗 Feature Correlation Heatmap")
    st.markdown("""
    <div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 15px;">
        This heatmap shows how strongly each feature is related to the final grade (G3). 
        Positive values (green) indicate features that help improve grades. 
        Negative values (red) indicate features that may harm performance.
    </div>
    """, unsafe_allow_html=True)
    
    # Select only numerical columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale=["#FF4B4B", "#1a1a2e", "#00C896"],  # Red → Dark → Green
        aspect="auto",
        zmin=-1, zmax=1,
    )
    fig_corr.update_layout(
        **CHART_THEME,
        height=600,
        xaxis_tickangle=-45,  # Angle x-axis labels for readability
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # --- Top correlations with G3 ---
    st.markdown("#### 🏆 Top Features Correlated with Final Grade (G3)")
    g3_corr = corr_matrix["G3"].drop("G3").sort_values(ascending=False)
    
    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("**✅ Positive Correlations (Higher value → Better grade)**")
        top_positive = g3_corr.head(5)
        for feat, val in top_positive.items():
            bar_width = int(abs(val) * 100)
            color = "#00C896"
            st.markdown(
                f'<div style="margin: 5px 0;">'
                f'<span style="color: #ccc; width: 120px; display: inline-block;">{feat}</span>'
                f'<div style="display: inline-block; background: {color}; width: {bar_width}px; height: 12px; border-radius: 6px; margin-right: 8px;"></div>'
                f'<span style="color: #a0a0b0; font-size: 0.8rem;">{val:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    with col_bot:
        st.markdown("**❌ Negative Correlations (Higher value → Lower grade)**")
        top_negative = g3_corr.tail(5)
        for feat, val in top_negative.items():
            bar_width = int(abs(val) * 100)
            color = "#FF4B4B"
            st.markdown(
                f'<div style="margin: 5px 0;">'
                f'<span style="color: #ccc; width: 120px; display: inline-block;">{feat}</span>'
                f'<div style="display: inline-block; background: {color}; width: {bar_width}px; height: 12px; border-radius: 6px; margin-right: 8px;"></div>'
                f'<span style="color: #a0a0b0; font-size: 0.8rem;">{val:.3f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

# ============================================================
# SECONDARY DATASET 1: Performance Index
# ============================================================
elif dataset_choice == "Performance Index" and df_secondary1 is not None:
    df = df_secondary1.copy()
    df = df.dropna(subset=["Performance Index"])  # Remove rows with missing target
    
    st.markdown("### 📈 Student Performance Index Analysis")
    st.markdown("""
    <div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 20px;">
        This dataset (10,000 students) tracks the relationship between study habits 
        and a Performance Index score.
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Students", f"{len(df):,}")
    with m2:
        st.metric("Avg Performance", f"{df['Performance Index'].mean():.1f}")
    with m3:
        st.metric("Avg Hours Studied", f"{df['Hours Studied'].mean():.1f}")
    with m4:
        st.metric("Avg Sleep Hours", f"{df['Sleep Hours'].mean():.1f}")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter: Hours Studied vs Performance
        fig = px.scatter(
            df, x="Hours Studied", y="Performance Index",
            color="Extracurricular Activities",
            title="📚 Hours Studied vs Performance Index",
            color_discrete_sequence=["#FF6B6B", "#00C896"],
            opacity=0.5,
        )
        # Manual trend line using numpy polyfit (avoids statsmodels dependency)
        z = np.polyfit(df["Hours Studied"], df["Performance Index"], 1)
        x_line = np.linspace(df["Hours Studied"].min(), df["Hours Studied"].max(), 100)
        y_line = np.polyval(z, x_line)
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color="#FFC857", width=2, dash="dash"),
        ))
        fig.update_layout(**CHART_THEME)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter: Sleep Hours vs Performance
        fig = px.scatter(
            df, x="Sleep Hours", y="Performance Index",
            color="Extracurricular Activities",
            title="😴 Sleep Hours vs Performance Index",
            color_discrete_sequence=["#FF6B6B", "#00C896"],
            opacity=0.5,
        )
        fig.update_layout(**CHART_THEME)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of Performance Index
    fig_dist = px.histogram(
        df, x="Performance Index",
        nbins=30,
        title="📊 Performance Index Distribution",
        color_discrete_sequence=["#00B4D8"],
        opacity=0.8,
    )
    fig_dist.update_layout(**CHART_THEME)
    st.plotly_chart(fig_dist, use_container_width=True)

# ============================================================
# SECONDARY DATASET 2: Exam Scores
# ============================================================
elif dataset_choice == "Exam Scores" and df_secondary2 is not None:
    df = df_secondary2.copy()
    
    st.markdown("### 📝 Student Exam Scores Analysis")
    st.markdown("""
    <div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 20px;">
        This dataset (1,000 students) contains math, reading, and writing scores 
        along with demographic information.
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Students", f"{len(df):,}")
    with m2:
        st.metric("Avg Math Score", f"{df['math score'].mean():.1f}")
    with m3:
        st.metric("Avg Reading Score", f"{df['reading score'].mean():.1f}")
    with m4:
        st.metric("Avg Writing Score", f"{df['writing score'].mean():.1f}")
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot: Scores by Gender
        score_data = df.melt(
            id_vars=["gender"],
            value_vars=["math score", "reading score", "writing score"],
            var_name="Subject",
            value_name="Score"
        )
        fig = px.box(
            score_data, x="Subject", y="Score",
            color="gender",
            title="📊 Score Distribution by Gender",
            color_discrete_map={"male": "#00B4D8", "female": "#FF6B6B"},
        )
        fig.update_layout(**CHART_THEME)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart: Average scores by test preparation
        prep_avg = df.groupby("test preparation course")[
            ["math score", "reading score", "writing score"]
        ].mean().reset_index()
        prep_melt = prep_avg.melt(
            id_vars=["test preparation course"],
            var_name="Subject",
            value_name="Average Score"
        )
        fig = px.bar(
            prep_melt, x="Subject", y="Average Score",
            color="test preparation course",
            title="📝 Impact of Test Preparation",
            barmode="group",
            color_discrete_sequence=["#FF6B6B", "#00C896"],
        )
        fig.update_layout(**CHART_THEME)
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter: Math vs Reading colored by Writing
    fig = px.scatter(
        df, x="math score", y="reading score",
        color="writing score",
        title="🔗 Math vs Reading Score (colored by Writing Score)",
        color_continuous_scale=["#FF4B4B", "#FFC857", "#00C896"],
        opacity=0.6,
    )
    fig.update_layout(**CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

else:
    # No data loaded
    st.warning("⚠️ Could not load the selected dataset. Please check if the data files exist.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-divider"></div>
<div style="text-align: center; color: #555580; font-size: 0.75rem; padding-bottom: 20px;">
    Dashboard — AI Student Performance Prediction System
</div>
""", unsafe_allow_html=True)

"""
app.py - Main Streamlit Application Entry Point.

This is the HOME PAGE of the AI Student Performance Prediction System.
It provides:
1. Welcome screen with project overview
2. Navigation to all sub-pages via sidebar
3. Quick statistics overview
4. System status indicators

Run this application with:
    streamlit run app.py

The sidebar automatically discovers pages in the 'pages/' directory.
Each page file is named with a number prefix for ordering (e.g., 1_Dashboard.py).
"""

import streamlit as st
import os
import sys
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_preprocessing import load_primary_dataset

# ============================================================
# PAGE CONFIGURATION
# ============================================================
# This must be the first Streamlit command in the script
st.set_page_config(
    page_title=config.APP_TITLE,            # Browser tab title
    page_icon=config.APP_ICON,              # Browser tab icon
    layout=config.APP_LAYOUT,               # "wide" for full-width layout
    initial_sidebar_state="expanded",        # Sidebar starts open
    menu_items={
        "About": "AI-Based Student Performance Prediction & Academic Risk Analysis System"
    }
)

# ============================================================
# CUSTOM CSS STYLING
# ============================================================
# Streamlit allows custom CSS injection for advanced styling
# We use this to create a premium, modern look
st.markdown("""
<style>
    /* ---- IMPORT GOOGLE FONT ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ---- GLOBAL STYLES ---- */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* ---- MAIN BACKGROUND GRADIENT ---- */
    .stApp {
        background: linear-gradient(135deg, #0F0C29 0%, #302B63 50%, #24243e 100%);
    }
    
    /* ---- SIDEBAR STYLING ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #a0a0b0;
    }
    
    /* ---- METRIC CARDS ---- */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 32px rgba(0, 200, 150, 0.15);
    }
    
    [data-testid="stMetric"] label {
        color: #a0a0b0 !important;
        font-size: 0.85rem;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    /* ---- GLASSMORPHISM CARDS ---- */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 200, 150, 0.3);
        box-shadow: 0 8px 32px rgba(0, 200, 150, 0.1);
    }
    
    /* ---- HERO SECTION ---- */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00C896 0%, #00B4D8 50%, #90E0EF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 10px;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a0a0b0;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 300;
    }
    
    /* ---- FEATURE CARDS ---- */
    .feature-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(0, 200, 150, 0.3);
        transform: translateY(-3px);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 12px;
    }
    
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 8px;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #8888a0;
        line-height: 1.5;
    }
    
    /* ---- STATUS BADGE ---- */
    .status-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px;
    }
    
    .status-active {
        background: rgba(0, 200, 150, 0.15);
        color: #00C896;
        border: 1px solid rgba(0, 200, 150, 0.3);
    }
    
    .status-inactive {
        background: rgba(255, 75, 75, 0.15);
        color: #FF4B4B;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    /* ---- BUTTONS ---- */
    .stButton > button {
        background: linear-gradient(135deg, #00C896 0%, #00B4D8 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 200, 150, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 200, 150, 0.4);
    }
    
    /* ---- DIVIDER ---- */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,200,150,0.3), transparent);
        margin: 30px 0;
    }
    
    /* ---- SCROLLBAR ---- */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #302B63;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #00C896;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HERO SECTION
# ============================================================
# The main welcome banner with project title and description
st.markdown("""
<div style="text-align: center; padding: 40px 0 20px 0;">
    <div class="hero-title">🎓 AI Student Performance Predictor</div>
    <div class="hero-subtitle">
        Intelligent Academic Analytics Platform — Predict Performance • Identify Risk • Generate Recommendations
    </div>
</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# SYSTEM STATUS CHECK
# ============================================================
# Check if the ML model is trained and ready
model_exists = os.path.exists(os.path.join(config.MODEL_SAVE_DIR, "best_model.pkl"))
dataset_exists = os.path.exists(config.PRIMARY_DATASET)

# Display status badges
col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    if dataset_exists:
        st.markdown(
            '<span class="status-badge status-active">✅ Dataset Loaded</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-badge status-inactive">❌ Dataset Missing</span>',
            unsafe_allow_html=True
        )

with col_status2:
    if model_exists:
        st.markdown(
            '<span class="status-badge status-active">✅ Model Trained</span>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-badge status-inactive">⚠️ Model Not Trained</span>',
            unsafe_allow_html=True
        )

with col_status3:
    st.markdown(
        '<span class="status-badge status-active">✅ Risk Advisor Active</span>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# QUICK STATS (if dataset is available)
# ============================================================
if dataset_exists:
    df = load_primary_dataset()
    if df is not None:
        # Display key statistics about the loaded dataset
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.metric(
                label="Total Students",      # Metric label
                value=f"{len(df):,}",         # Total number of students
                delta="Primary Dataset"       # Subtitle
            )
        
        with c2:
            pass_count = (df["G3"] >= config.PASS_THRESHOLD).sum()
            st.metric(
                label="Passing Students",
                value=f"{pass_count}",
                delta=f"{pass_count/len(df)*100:.1f}%"  # Pass percentage
            )
        
        with c3:
            fail_count = (df["G3"] < config.PASS_THRESHOLD).sum()
            st.metric(
                label="Failing Students",
                value=f"{fail_count}",
                delta=f"{fail_count/len(df)*100:.1f}%"  # Fail percentage
            )
        
        with c4:
            st.metric(
                label="Avg Grade",
                value=f"{df['G3'].mean():.1f}/20",  # Average final grade
                delta=f"Median: {df['G3'].median():.0f}"
            )

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# FEATURE CARDS
# ============================================================
# Display the key features of the system in a grid layout
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; padding: 10px;">
    <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">ML Prediction</div>
        <div class="feature-desc">3 algorithms compared:<br>Logistic Regression, Decision Tree, Random Forest</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Analytics Dashboard</div>
        <div class="feature-desc">Interactive charts and visualizations for deep academic insights</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">⚠️</div>
        <div class="feature-title">Risk Classification</div>
        <div class="feature-desc">Automatically categorize students into High, Medium, or Low risk levels</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI Risk Advisor</div>
        <div class="feature-desc">Intelligent agent generates personalized academic recommendations</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📁</div>
        <div class="feature-title">Batch Upload</div>
        <div class="feature-desc">Upload CSV files to predict performance for entire classes at once</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📜</div>
        <div class="feature-title">Prediction History</div>
        <div class="feature-desc">Track all past predictions with timestamps and risk assessments</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ============================================================
# GETTING STARTED GUIDE
# ============================================================
st.markdown("""
<div class="glass-card">
    <h3 style="color: #00C896; margin-bottom: 15px;">🚀 Getting Started</h3>
    <p style="color: #c0c0d0; line-height: 1.8; font-size: 0.95rem;">
        <strong style="color: #00B4D8;">Step 1:</strong> Navigate to <strong>Model Training</strong> page to train the ML model on the student dataset.<br>
        <strong style="color: #00B4D8;">Step 2:</strong> Visit the <strong>Dashboard</strong> to explore data visualizations and analytics.<br>
        <strong style="color: #00B4D8;">Step 3:</strong> Go to <strong>Predict</strong> page to predict performance for individual students.<br>
        <strong style="color: #00B4D8;">Step 4:</strong> Use <strong>Batch Upload</strong> to process entire class CSV files at once.<br>
        <strong style="color: #00B4D8;">Step 5:</strong> Check <strong>History</strong> to review all past predictions and trends.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONTENT
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <div style="font-size: 2.5rem;">🎓</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #00C896; margin-top: 8px;">
            AI Performance Predictor
        </div>
        <div style="font-size: 0.75rem; color: #8888a0; margin-top: 4px;">
            Academic Risk Analysis System
        </div>
    </div>
    <div class="custom-divider"></div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0 10px;">
        <p style="color: #a0a0b0; font-size: 0.8rem; line-height: 1.6;">
            📌 <strong style="color: #ccc;">Project:</strong> Mini Project - 3rd Year<br>
            📌 <strong style="color: #ccc;">Domain:</strong> AI & Educational Analytics<br>
            📌 <strong style="color: #ccc;">Stack:</strong> Streamlit + Scikit-Learn + Supabase
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px 0; color: #555580; font-size: 0.75rem;">
    AI-Based Student Performance Prediction & Academic Risk Analysis System<br>
    Built with ❤️ using Streamlit, Scikit-Learn & Python
</div>
""", unsafe_allow_html=True)

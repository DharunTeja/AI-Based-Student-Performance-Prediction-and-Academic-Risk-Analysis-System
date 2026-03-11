"""
4_🤖_Model_Training.py - Model Training & Comparison Page.

This page allows faculty/admins to:
1. View the training dataset statistics
2. Train all 3 ML models (Logistic Regression, Decision Tree, Random Forest)
3. Compare model performance metrics side by side
4. View confusion matrices for each model
5. Visualize feature importance (which features matter most)
6. Select and save the best model

This is the core ML training interface of the system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_preprocessing import (
    load_primary_dataset,
    preprocess_primary_dataset,
    prepare_training_data,
)
from models.train_model import (
    train_all_models,
    select_best_model,
    save_model,
)
from utils.db_manager import save_model_metrics

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Model Training | Student Performance",
    page_icon="🤖",
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
        box-shadow: 0 4px 15px rgba(0, 200, 150, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 200, 150, 0.4);
    }
    .model-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border-color: rgba(0,200,150,0.3);
        transform: translateY(-3px);
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
<div class="page-title">🤖 Model Training & Comparison</div>
<div class="page-subtitle">Train ML models, compare performance, and select the best classifier</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# CHART THEME
# ============================================================
CHART_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#a0a0b0", "family": "Inter"},
    "title_font": {"color": "#ffffff", "size": 16},
    "xaxis": {"gridcolor": "rgba(255,255,255,0.05)"},
    "yaxis": {"gridcolor": "rgba(255,255,255,0.05)"},
}

# ============================================================
# LOAD AND PREVIEW DATASET
# ============================================================
st.markdown("### 📁 Training Dataset")

df = load_primary_dataset()
if df is None:
    st.error("❌ Could not load the primary dataset. Please check the file path.")
    st.stop()

# Dataset stats
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Records", f"{len(df):,}")
with m2:
    st.metric("Features", f"{len(df.columns)}")
with m3:
    pass_count = (df["G3"] >= config.PASS_THRESHOLD).sum()
    st.metric("Pass", f"{pass_count} ({pass_count/len(df)*100:.0f}%)")
with m4:
    fail_count = (df["G3"] < config.PASS_THRESHOLD).sum()
    st.metric("Fail", f"{fail_count} ({fail_count/len(df)*100:.0f}%)")
with m5:
    st.metric("Avg Grade", f"{df['G3'].mean():.1f}/20")

# Data preview
with st.expander("👀 Preview Dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True, height=300)
    
    st.markdown("#### 📊 Column Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ============================================================
# TRAINING SECTION
# ============================================================
st.markdown("### 🚀 Train Machine Learning Models")

st.markdown("""
<div class="glass-card">
    <p style="color: #a0a0b0; font-size: 0.9rem; line-height: 1.8;">
        Click the button below to start training <strong style="color: #00C896;">3 ML algorithms</strong>:<br>
        <strong style="color: #00B4D8;">1. Logistic Regression</strong> — Linear classifier, fast and interpretable<br>
        <strong style="color: #00B4D8;">2. Decision Tree</strong> — Rule-based classifier, easy to understand<br>
        <strong style="color: #00B4D8;">3. Random Forest</strong> — Ensemble method, usually highest accuracy<br><br>
        The system will automatically compare all models and save the best one.
    </p>
</div>
""", unsafe_allow_html=True)

# Training configuration in sidebar
with st.sidebar:
    st.markdown("### ⚙️ Training Config")
    test_size = st.slider(
        "Test Size (%)",
        min_value=10, max_value=40, value=20,
        help="Percentage of data used for testing (rest for training)"
    )
    random_state = st.number_input(
        "Random Seed",
        min_value=0, max_value=999, value=42,
        help="Seed for reproducibility"
    )

# TRAIN BUTTON
if st.button("🤖 Start Training", use_container_width=True, type="primary"):
    
    # Step 1: Preprocess data
    with st.spinner("🔧 Preprocessing data..."):
        progress = st.progress(0, text="Step 1/5: Preprocessing data...")
        X, y, feature_names, label_encoders = preprocess_primary_dataset(df.copy())
        time.sleep(0.5)
        progress.progress(20, text="Step 2/5: Splitting dataset...")
    
    # Step 2: Split data
    with st.spinner("✂️ Splitting dataset..."):
        X_train, X_test, y_train, y_test = prepare_training_data(X, y)
        time.sleep(0.3)
        progress.progress(40, text="Step 3/5: Training models...")
    
    st.markdown(f"""
    <div style="color: #a0a0b0; font-size: 0.85rem; margin: 10px 0;">
        📊 Training set: <strong style="color: #00C896;">{len(X_train)}</strong> records | 
        Testing set: <strong style="color: #00B4D8;">{len(X_test)}</strong> records |
        Features: <strong style="color: #FFC857;">{len(feature_names)}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3: Train all models
    with st.spinner("🤖 Training 3 ML models... This may take a moment."):
        results = train_all_models(X_train, y_train, X_test, y_test)
        progress.progress(70, text="Step 4/5: Evaluating models...")
    
    # Step 4: Select best model
    with st.spinner("🏆 Selecting best model..."):
        best_name, best_result = select_best_model(results)
        time.sleep(0.3)
        progress.progress(90, text="Step 5/5: Saving model...")
    
    # Step 5: Save best model
    with st.spinner("💾 Saving best model..."):
        save_model(
            model=best_result["model"],
            scaler=best_result["scaler"],
            feature_names=feature_names,
            label_encoders=label_encoders,
            model_name="best_model"
        )
        progress.progress(100, text="✅ Training complete!")
    
    # Save metrics for each model
    for name, result in results.items():
        save_model_metrics(name, result["metrics"])
    
    st.success(f"🏆 Training complete! Best model: **{best_name}** (Accuracy: {best_result['metrics']['accuracy']:.1%})")
    
    # Store results in session state so they persist after rerun
    st.session_state["training_results"] = results
    st.session_state["best_model_name"] = best_name
    st.session_state["feature_names"] = feature_names

# ============================================================
# DISPLAY RESULTS (if training has been done)
# ============================================================
if "training_results" in st.session_state:
    results = st.session_state["training_results"]
    best_name = st.session_state["best_model_name"]
    feature_names = st.session_state["feature_names"]
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # --- MODEL COMPARISON TABLE ---
    st.markdown("### 📊 Model Performance Comparison")
    
    # Create comparison data
    comparison_data = []
    for name, result in results.items():
        metrics = result["metrics"]
        comparison_data.append({
            "Model": name,
            "Accuracy": f"{metrics['accuracy']:.1%}",
            "Precision": f"{metrics['precision']:.1%}",
            "Recall": f"{metrics['recall']:.1%}",
            "F1 Score": f"{metrics['f1_score']:.1%}",
            "Best": "🏆" if name == best_name else "",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # --- METRICS BAR CHART ---
    st.markdown("### 📈 Visual Comparison")
    
    # Grouped bar chart comparing all metrics across models
    metrics_data = []
    for name, result in results.items():
        for metric, value in result["metrics"].items():
            metrics_data.append({
                "Model": name,
                "Metric": metric.replace("_", " ").title(),
                "Value": value,
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    fig_compare = px.bar(
        metrics_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",              # Group bars side by side
        title="📊 Model Performance Metrics Comparison",
        color_discrete_sequence=["#00C896", "#00B4D8", "#FFC857"],
        text=metrics_df["Value"].apply(lambda x: f"{x:.1%}"),
    )
    fig_compare.update_layout(**CHART_THEME, height=450)
    fig_compare.update_traces(textposition="outside", textfont_size=11)
    fig_compare.update_yaxes(range=[0, 1.15])
    st.plotly_chart(fig_compare, use_container_width=True)
    
    # --- CONFUSION MATRICES ---
    st.markdown("### 🔲 Confusion Matrices")
    st.markdown("""
    <div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 15px;">
        Confusion matrices show how the model's predictions compare to actual results.<br>
        <strong>True Positive:</strong> Correctly predicted Pass | 
        <strong>True Negative:</strong> Correctly predicted Fail<br>
        <strong>False Positive:</strong> Predicted Pass but actually Fail | 
        <strong>False Negative:</strong> Predicted Fail but actually Pass
    </div>
    """, unsafe_allow_html=True)
    
    cm_cols = st.columns(3)
    model_colors = {
        "Logistic Regression": ["#1a1a2e", "#00C896"],
        "Decision Tree": ["#1a1a2e", "#00B4D8"],
        "Random Forest": ["#1a1a2e", "#FFC857"],
    }
    
    for idx, (name, result) in enumerate(results.items()):
        with cm_cols[idx]:
            cm = result["confusion_matrix"]
            labels = ["Fail (0)", "Pass (1)"]
            
            fig_cm = px.imshow(
                cm,
                x=labels,
                y=labels,
                title=f"{name}",
                color_continuous_scale=model_colors[name],
                text_auto=True,             # Show numbers on cells
                labels={"x": "Predicted", "y": "Actual"},
            )
            fig_cm.update_layout(
                **CHART_THEME,
                height=350,
                title_font_size=14,
            )
            fig_cm.update_traces(textfont_size=18)
            st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # --- FEATURE IMPORTANCE (for Random Forest) ---
    st.markdown("### 🎯 Feature Importance (Random Forest)")
    st.markdown("""
    <div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 15px;">
        This chart shows which features have the most influence on the prediction.
        Higher values mean the feature is more important for predicting student performance.
    </div>
    """, unsafe_allow_html=True)
    
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["model"]
        # Get feature importance from the Random Forest model
        importances = rf_model.feature_importances_
        
        # Create a sorted DataFrame of feature importances
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=True)  # Sort ascending for horizontal bar
        
        # Show top 15 most important features
        feat_imp_top = feat_imp.tail(15)
        
        fig_imp = px.bar(
            feat_imp_top,
            x="Importance",
            y="Feature",
            orientation="h",          # Horizontal bars
            title="Top 15 Most Important Features",
            color="Importance",
            color_continuous_scale=["#302B63", "#00C896"],
            text=feat_imp_top["Importance"].apply(lambda x: f"{x:.3f}"),
        )
        fig_imp.update_layout(**CHART_THEME, height=500)
        fig_imp.update_traces(textposition="outside")
        st.plotly_chart(fig_imp, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-divider"></div>
<div style="text-align: center; color: #555580; font-size: 0.75rem; padding-bottom: 20px;">
    Model Training — AI Student Performance Prediction System
</div>
""", unsafe_allow_html=True)

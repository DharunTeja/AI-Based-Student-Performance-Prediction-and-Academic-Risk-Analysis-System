"""
3_📁_Batch_Upload.py - Batch CSV Upload & Prediction Page.

This page allows faculty to:
1. Upload a CSV file containing multiple student records
2. Preview the uploaded data
3. Run batch predictions on all students at once
4. View results with risk level distribution
5. Download the results as a CSV file
6. Save all predictions to history

This is useful for processing entire class lists at once.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import io  # For creating downloadable CSV files

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.predictor import load_model, predict_batch
from utils.db_manager import save_prediction
from utils.data_preprocessing import get_risk_level

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Batch Upload | Student Performance",
    page_icon="📁",
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-title">📁 Batch Upload & Predict</div>
<div class="page-subtitle">Upload a CSV file with multiple student records for batch prediction</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# CHECK MODEL
# ============================================================
model_bundle = load_model()

if model_bundle is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 3rem; margin-bottom: 15px;">⚠️</div>
        <h3 style="color: #FFC857;">Model Not Trained Yet</h3>
        <p style="color: #a0a0b0;">
            Please go to the <strong style="color: #00C896;">Model Training</strong> page first 
            to train the ML model before making predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================
# FILE UPLOAD SECTION
# ============================================================
st.markdown("""
<div class="glass-card">
    <h4 style="color: #00C896; margin-bottom: 10px;">📤 Upload Student Data</h4>
    <p style="color: #a0a0b0; font-size: 0.85rem;">
        Upload a CSV file with the same format as the training dataset (student-mat.csv).<br>
        The file should contain columns like: school, sex, age, studytime, G1, G2, absences, etc.<br>
        <strong style="color: #00B4D8;">Note:</strong> The CSV file should use semicolon (;) or comma (,) as separator.
    </p>
</div>
""", unsafe_allow_html=True)

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],                    # Only accept CSV files
    help="Upload a CSV file with student data",
    label_visibility="collapsed",     # Hide the default label
)

# ============================================================
# SAMPLE DATA DOWNLOAD
# ============================================================
# Provide a sample CSV for users to see the expected format
with st.expander("📋 View Expected CSV Format & Download Sample"):
    st.markdown("""
    <p style="color: #a0a0b0; font-size: 0.85rem;">
        Your CSV file should have these columns (in any order):
    </p>
    """, unsafe_allow_html=True)
    
    # Create a sample DataFrame with expected columns
    sample_data = pd.DataFrame({
        "school": ["GP", "MS"],
        "sex": ["F", "M"],
        "age": [18, 17],
        "address": ["U", "R"],
        "famsize": ["GT3", "GT3"],
        "Pstatus": ["A", "T"],
        "Medu": [4, 1],
        "Fedu": [4, 1],
        "Mjob": ["at_home", "at_home"],
        "Fjob": ["teacher", "other"],
        "reason": ["course", "course"],
        "guardian": ["mother", "father"],
        "traveltime": [2, 1],
        "studytime": [2, 2],
        "failures": [0, 3],
        "schoolsup": ["yes", "no"],
        "famsup": ["no", "yes"],
        "paid": ["no", "no"],
        "activities": ["no", "no"],
        "nursery": ["yes", "yes"],
        "higher": ["yes", "yes"],
        "internet": ["no", "yes"],
        "romantic": ["no", "no"],
        "famrel": [4, 5],
        "freetime": [3, 5],
        "goout": [4, 4],
        "Dalc": [1, 2],
        "Walc": [1, 2],
        "health": [3, 3],
        "absences": [6, 4],
        "G1": [5, 5],
        "G2": [6, 5],
    })
    
    st.dataframe(sample_data, use_container_width=True)
    
    # Download button for sample CSV
    csv_sample = sample_data.to_csv(index=False, sep=";")
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv_sample,
        file_name="sample_student_data.csv",
        mime="text/csv",
    )

# ============================================================
# PROCESS UPLOADED FILE
# ============================================================
if uploaded_file is not None:
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    
    # Try reading with different separators
    try:
        # First try semicolon separator (used in student-mat.csv)
        df_uploaded = pd.read_csv(uploaded_file, delimiter=";")
        if len(df_uploaded.columns) <= 2:
            # If only 1-2 columns, probably comma-separated
            uploaded_file.seek(0)  # Reset file pointer
            df_uploaded = pd.read_csv(uploaded_file, delimiter=",")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        st.stop()
    
    # --- PREVIEW DATA ---
    st.markdown("### 👀 Data Preview")
    st.markdown(f"""
    <div style="color: #a0a0b0; font-size: 0.85rem; margin-bottom: 10px;">
        Loaded <strong style="color: #00C896;">{len(df_uploaded)}</strong> students with 
        <strong style="color: #00B4D8;">{len(df_uploaded.columns)}</strong> columns
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(
        df_uploaded.head(10),          # Show first 10 rows
        use_container_width=True,
        height=300,
    )
    
    # --- PREDICT BUTTON ---
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮 Run Batch Prediction", use_container_width=True, type="primary"):
        
        # Show progress bar during prediction
        progress_bar = st.progress(0, text="Analyzing student data...")
        
        with st.spinner("🤖 AI is processing all students..."):
            # Run batch predictions
            results_df = predict_batch(df_uploaded, model_bundle)
            progress_bar.progress(100, text="✅ Complete!")
        
        if results_df is not None:
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # --- BATCH RESULTS SUMMARY ---
            st.markdown("### 📊 Batch Prediction Results")
            
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            
            total = len(results_df)
            pass_count = (results_df["prediction"] == "Pass").sum()
            fail_count = (results_df["prediction"] == "Fail").sum()
            
            with m1:
                st.metric("Total Students", f"{total}")
            with m2:
                st.metric("Predicted Pass", f"{pass_count}", delta=f"{pass_count/total*100:.1f}%")
            with m3:
                st.metric("Predicted Fail", f"{fail_count}", delta=f"-{fail_count/total*100:.1f}%")
            with m4:
                high_risk = (results_df["risk_level"] == "High Risk").sum()
                st.metric("High Risk", f"{high_risk}", delta=f"{high_risk/total*100:.1f}%")
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # --- CHARTS ---
            col1, col2 = st.columns(2)
            
            with col1:
                # Prediction distribution pie chart
                pred_counts = results_df["prediction"].value_counts()
                fig_pred = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="🎯 Prediction Distribution",
                    color=pred_counts.index,
                    color_discrete_map={"Pass": "#00C896", "Fail": "#FF4B4B"},
                    hole=0.4,
                )
                fig_pred.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#a0a0b0"},
                )
                fig_pred.update_traces(textinfo="label+percent+value")
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                # Risk level distribution bar chart
                risk_counts = results_df["risk_level"].value_counts().reset_index()
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
                fig_risk.update_traces(textposition="outside")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            # --- PROBABILITY DISTRIBUTION ---
            fig_prob = px.histogram(
                results_df,
                x="pass_probability",
                nbins=20,
                title="📊 Pass Probability Distribution",
                color_discrete_sequence=["#00B4D8"],
                opacity=0.8,
                labels={"pass_probability": "Pass Probability"},
            )
            fig_prob.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#a0a0b0"},
                xaxis={"gridcolor": "rgba(255,255,255,0.05)"},
                yaxis={"gridcolor": "rgba(255,255,255,0.05)"},
            )
            # Add risk threshold lines
            fig_prob.add_vline(x=0.4, line_dash="dash", line_color="#FF4B4B",
                             annotation_text="High Risk", annotation_font_color="#FF4B4B")
            fig_prob.add_vline(x=0.7, line_dash="dash", line_color="#FFC857",
                             annotation_text="Medium Risk", annotation_font_color="#FFC857")
            st.plotly_chart(fig_prob, use_container_width=True)
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # --- RESULTS TABLE ---
            st.markdown("### 📋 Detailed Results")
            
            # Color-code the risk level in the table
            st.dataframe(
                results_df,
                use_container_width=True,
                height=400,
            )
            
            # --- DOWNLOAD RESULTS ---
            st.markdown("<br>", unsafe_allow_html=True)
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_results,
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            
            # --- SAVE TO HISTORY ---
            # Save each prediction to the history database
            saved_count = 0
            for _, row in results_df.iterrows():
                student_data = row.drop(["prediction", "pass_probability", "risk_level"]).to_dict()
                save_prediction(
                    student_data=student_data,
                    prediction=1 if row["prediction"] == "Pass" else 0,
                    risk_level=row["risk_level"],
                    probability=row["pass_probability"],
                    recommendations=[]
                )
                saved_count += 1
            
            st.success(f"✅ {saved_count} predictions saved to history!")
        
        else:
            st.error("❌ Batch prediction failed. Please check the CSV format.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-divider"></div>
<div style="text-align: center; color: #555580; font-size: 0.75rem; padding-bottom: 20px;">
    Batch Upload — AI Student Performance Prediction System
</div>
""", unsafe_allow_html=True)

"""
2_🔮_Predict.py - Individual Student Prediction Page.

This page allows faculty to:
1. Enter individual student data through a form
2. Get real-time prediction (Pass/Fail)
3. View risk level classification
4. See personalized recommendations from the AI Risk Advisor
5. Save prediction to history

The form dynamically generates input fields based on the model's features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.predictor import load_model, predict_single_student
from utils.data_preprocessing import get_feature_descriptions, load_primary_dataset
from utils.db_manager import save_prediction

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Predict | Student Performance",
    page_icon="🔮",
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
    .risk-high {
        background: rgba(255, 75, 75, 0.15);
        border: 1px solid rgba(255, 75, 75, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .risk-medium {
        background: rgba(255, 165, 0, 0.15);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .risk-low {
        background: rgba(0, 200, 150, 0.15);
        border: 1px solid rgba(0, 200, 150, 0.3);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .recommendation-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    .recommendation-card:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(0,200,150,0.2);
    }
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# PAGE HEADER
# ============================================================
st.markdown("""
<div class="page-title">🔮 Predict Student Performance</div>
<div class="page-subtitle">Enter student details to get AI-powered performance prediction & risk assessment</div>
<div class="custom-divider"></div>
""", unsafe_allow_html=True)

# ============================================================
# CHECK IF MODEL IS TRAINED
# ============================================================
model_bundle = load_model()

if model_bundle is None:
    # Model not trained yet - show instructions
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
    st.stop()  # Stop execution here

# ============================================================
# STUDENT INPUT FORM
# ============================================================
st.markdown("### 📝 Student Information Form")
st.markdown("""
<div style="color: #8888a0; font-size: 0.85rem; margin-bottom: 15px;">
    Fill in the student's details below. All fields are required for accurate prediction.
</div>
""", unsafe_allow_html=True)

# Get feature descriptions for tooltips
feature_desc = get_feature_descriptions()

# Load dataset to get value ranges
df = load_primary_dataset()

# ============================================================
# FORM LAYOUT - Organized into sections
# ============================================================
with st.form("prediction_form"):
    
    # --- SECTION 1: Personal Information ---
    st.markdown("#### 👤 Personal Information")
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        school = st.selectbox(
            "School",
            options=["GP", "MS"],
            help=feature_desc.get("school", "")  # Shows tooltip on hover
        )
    with p2:
        sex = st.selectbox(
            "Sex",
            options=["M", "F"],
            help=feature_desc.get("sex", "")
        )
    with p3:
        age = st.slider(
            "Age",
            min_value=15, max_value=22, value=17,
            help=feature_desc.get("age", "")
        )
    with p4:
        address = st.selectbox(
            "Address Type",
            options=["U", "R"],  # Urban / Rural
            help=feature_desc.get("address", "")
        )
    
    # --- SECTION 2: Family Information ---
    st.markdown("#### 👨‍👩‍👧 Family Information")
    f1, f2, f3, f4 = st.columns(4)
    
    with f1:
        famsize = st.selectbox(
            "Family Size",
            options=["LE3", "GT3"],  # ≤3 or >3 members
            help=feature_desc.get("famsize", "")
        )
    with f2:
        pstatus = st.selectbox(
            "Parents' Status",
            options=["T", "A"],  # Together / Apart
            help=feature_desc.get("Pstatus", "")
        )
    with f3:
        medu = st.slider(
            "Mother's Education",
            min_value=0, max_value=4, value=2,
            help=feature_desc.get("Medu", "")
        )
    with f4:
        fedu = st.slider(
            "Father's Education",
            min_value=0, max_value=4, value=2,
            help=feature_desc.get("Fedu", "")
        )
    
    f5, f6, f7, f8 = st.columns(4)
    with f5:
        mjob = st.selectbox(
            "Mother's Job",
            options=["teacher", "health", "services", "at_home", "other"],
            help=feature_desc.get("Mjob", "")
        )
    with f6:
        fjob = st.selectbox(
            "Father's Job",
            options=["teacher", "health", "services", "at_home", "other"],
            help=feature_desc.get("Fjob", "")
        )
    with f7:
        guardian = st.selectbox(
            "Guardian",
            options=["mother", "father", "other"],
            help=feature_desc.get("guardian", "")
        )
    with f8:
        famrel = st.slider(
            "Family Relationship Quality",
            min_value=1, max_value=5, value=4,
            help=feature_desc.get("famrel", "")
        )
    
    # --- SECTION 3: Academic Information ---
    st.markdown("#### 📚 Academic Information")
    a1, a2, a3, a4 = st.columns(4)
    
    with a1:
        reason = st.selectbox(
            "Reason for School Choice",
            options=["home", "reputation", "course", "other"],
            help=feature_desc.get("reason", "")
        )
    with a2:
        traveltime = st.slider(
            "Travel Time",
            min_value=1, max_value=4, value=1,
            help=feature_desc.get("traveltime", "")
        )
    with a3:
        studytime = st.slider(
            "Weekly Study Time",
            min_value=1, max_value=4, value=2,
            help=feature_desc.get("studytime", "")
        )
    with a4:
        failures = st.slider(
            "Past Failures",
            min_value=0, max_value=4, value=0,
            help=feature_desc.get("failures", "")
        )
    
    a5, a6, a7, a8 = st.columns(4)
    with a5:
        schoolsup = st.selectbox(
            "School Support",
            options=["yes", "no"],
            help=feature_desc.get("schoolsup", "")
        )
    with a6:
        famsup = st.selectbox(
            "Family Support",
            options=["yes", "no"],
            help=feature_desc.get("famsup", "")
        )
    with a7:
        paid = st.selectbox(
            "Paid Classes",
            options=["no", "yes"],
            help=feature_desc.get("paid", "")
        )
    with a8:
        activities = st.selectbox(
            "Extracurricular",
            options=["yes", "no"],
            help=feature_desc.get("activities", "")
        )
    
    # --- SECTION 4: Grades & Absences ---
    st.markdown("#### 📊 Grades & Attendance")
    g1, g2, g3, g4 = st.columns(4)
    
    with g1:
        grade_g1 = st.slider(
            "First Period Grade (G1)",
            min_value=0, max_value=20, value=10,
            help=feature_desc.get("G1", "")
        )
    with g2:
        grade_g2 = st.slider(
            "Second Period Grade (G2)",
            min_value=0, max_value=20, value=10,
            help=feature_desc.get("G2", "")
        )
    with g3:
        absences = st.slider(
            "Number of Absences",
            min_value=0, max_value=93, value=5,
            help=feature_desc.get("absences", "")
        )
    with g4:
        health = st.slider(
            "Health Status",
            min_value=1, max_value=5, value=3,
            help=feature_desc.get("health", "")
        )
    
    # --- SECTION 5: Lifestyle ---
    st.markdown("#### 🎯 Lifestyle")
    l1, l2, l3, l4, l5, l6 = st.columns(6)
    
    with l1:
        nursery = st.selectbox("Nursery", ["yes", "no"], help=feature_desc.get("nursery", ""))
    with l2:
        higher = st.selectbox("Higher Education", ["yes", "no"], help=feature_desc.get("higher", ""))
    with l3:
        internet = st.selectbox("Internet", ["yes", "no"], help=feature_desc.get("internet", ""))
    with l4:
        romantic = st.selectbox("Romantic", ["no", "yes"], help=feature_desc.get("romantic", ""))
    with l5:
        freetime = st.slider("Free Time", 1, 5, 3, help=feature_desc.get("freetime", ""))
    with l6:
        goout = st.slider("Going Out", 1, 5, 3, help=feature_desc.get("goout", ""))
    
    l7, l8 = st.columns(2)
    with l7:
        dalc = st.slider("Workday Alcohol", 1, 5, 1, help=feature_desc.get("Dalc", ""))
    with l8:
        walc = st.slider("Weekend Alcohol", 1, 5, 1, help=feature_desc.get("Walc", ""))
    
    # --- SUBMIT BUTTON ---
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔮 Predict Performance",
        use_container_width=True,
    )

# ============================================================
# PREDICTION RESULTS
# ============================================================
if submitted:
    # Build the student data dictionary from form inputs
    student_data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": pstatus,
        "Medu": medu,
        "Fedu": fedu,
        "Mjob": mjob,
        "Fjob": fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": dalc,
        "Walc": walc,
        "health": health,
        "absences": absences,
        "G1": grade_g1,
        "G2": grade_g2,
    }
    
    # Show a spinner while the model is predicting
    with st.spinner("🤖 AI is analyzing student data..."):
        result = predict_single_student(student_data, model_bundle)
    
    if result:
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Results")
        
        # --- RESULT CARDS ---
        r1, r2, r3 = st.columns(3)
        
        with r1:
            # Prediction result (Pass/Fail)
            pred_color = "#00C896" if result["prediction"] == 1 else "#FF4B4B"
            pred_icon = "✅" if result["prediction"] == 1 else "❌"
            st.markdown(f"""
            <div class="glass-card" style="text-align: center;">
                <div style="font-size: 2.5rem;">{pred_icon}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {pred_color}; margin: 10px 0;">
                    {result["prediction_label"]}
                </div>
                <div style="color: #8888a0; font-size: 0.85rem;">Predicted Result</div>
            </div>
            """, unsafe_allow_html=True)
        
        with r2:
            # Risk Level
            risk = result["risk_level"]
            risk_class = "risk-high" if "High" in risk else ("risk-medium" if "Medium" in risk else "risk-low")
            risk_icon = "🔴" if "High" in risk else ("🟡" if "Medium" in risk else "🟢")
            st.markdown(f"""
            <div class="{risk_class}">
                <div style="font-size: 2.5rem;">{risk_icon}</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: white; margin: 10px 0;">
                    {risk}
                </div>
                <div style="color: #a0a0b0; font-size: 0.85rem;">Academic Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with r3:
            # Pass Probability Gauge
            prob = result["pass_probability"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": "white", "size": 40}},
                title={"text": "Pass Probability", "font": {"color": "#a0a0b0", "size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#555"},
                    "bar": {"color": "#00C896" if prob >= 0.7 else ("#FFC857" if prob >= 0.4 else "#FF4B4B")},
                    "bgcolor": "rgba(255,255,255,0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 40], "color": "rgba(255,75,75,0.1)"},
                        {"range": [40, 70], "color": "rgba(255,200,87,0.1)"},
                        {"range": [70, 100], "color": "rgba(0,200,150,0.1)"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 2},
                        "thickness": 0.75,
                        "value": prob * 100,
                    },
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#a0a0b0"},
                height=250,
                margin=dict(t=60, b=20, l=30, r=30),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
        
        # --- RECOMMENDATIONS FROM AI RISK ADVISOR ---
        st.markdown("### 🧠 AI Risk Advisor Recommendations")
        
        # Summary badges
        summary = result["risk_summary"]
        st.markdown(f"""
        <div style="display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap;">
            <div style="background: rgba(255,75,75,0.15); border: 1px solid rgba(255,75,75,0.3); 
                        border-radius: 10px; padding: 8px 16px;">
                <span style="color: #FF4B4B; font-weight: 700;">🚨 {summary['critical']}</span>
                <span style="color: #a0a0b0;"> Critical</span>
            </div>
            <div style="background: rgba(255,165,0,0.15); border: 1px solid rgba(255,165,0,0.3); 
                        border-radius: 10px; padding: 8px 16px;">
                <span style="color: #FFA500; font-weight: 700;">⚠️ {summary['important']}</span>
                <span style="color: #a0a0b0;"> Important</span>
            </div>
            <div style="background: rgba(0,200,150,0.15); border: 1px solid rgba(0,200,150,0.3); 
                        border-radius: 10px; padding: 8px 16px;">
                <span style="color: #00C896; font-weight: 700;">💡 {summary['suggested']}</span>
                <span style="color: #a0a0b0;"> Suggested</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display each recommendation
        for rec in result["recommendations"]:
            # Set color based on priority
            if rec["priority"] == "Critical":
                border_color = "rgba(255,75,75,0.3)"
                badge_bg = "rgba(255,75,75,0.15)"
                badge_color = "#FF4B4B"
            elif rec["priority"] == "Important":
                border_color = "rgba(255,165,0,0.3)"
                badge_bg = "rgba(255,165,0,0.15)"
                badge_color = "#FFA500"
            else:
                border_color = "rgba(0,200,150,0.3)"
                badge_bg = "rgba(0,200,150,0.15)"
                badge_color = "#00C896"
            
            st.markdown(f"""
            <div class="recommendation-card" style="border-color: {border_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 600; color: #ccc;">
                        {rec["icon"]} {rec["category"]}
                    </span>
                    <span style="background: {badge_bg}; color: {badge_color}; 
                                 padding: 3px 12px; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">
                        {rec["priority"]}
                    </span>
                </div>
                <p style="color: #a0a0b0; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                    {rec["recommendation"]}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # --- SAVE PREDICTION ---
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Save to database/local storage
        save_prediction(
            student_data=student_data,
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            probability=result["pass_probability"],
            recommendations=result["recommendations"]
        )
        st.success("✅ Prediction saved to history!")
    
    else:
        st.error("❌ Prediction failed. Please check the model and try again.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="custom-divider"></div>
<div style="text-align: center; color: #555580; font-size: 0.75rem; padding-bottom: 20px;">
    Student Prediction — AI Student Performance Prediction System
</div>
""", unsafe_allow_html=True)

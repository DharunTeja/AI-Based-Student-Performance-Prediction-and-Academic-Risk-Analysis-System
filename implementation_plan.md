# AI-Based Student Performance Prediction - Implementation Plan

## Tech Stack
-------------------------------------------------------------------------
|       Layer   |                       Technology                      |
|---------------|-------------------------------------------------------|
| **Frontend**  | Streamlit (Python-based web framework)                |
| **Backend**   | Python (pandas, scikit-learn, numpy, plotly)          |
| **Database**  | Supabase (PostgreSQL-based, free tier, easy setup)    |
| **ML Models** | Logistic Regression, Decision Tree, Random Forest     |
-------------------------------------------------------------------------

## Project Structure
```
Mini-Project_3rd Year/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration (DB credentials, constants)
├── models/
│   ├── train_model.py        # ML model training pipeline
│   ├── predictor.py          # Prediction logic
│   └── saved_models/         # Saved .pkl model files
├── utils/
│   ├── data_preprocessing.py # Data cleaning & feature engineering
│   ├── risk_advisor.py       # Academic Risk Advisor Agent
│   └── db_manager.py         # Database operations (Supabase)
├── pages/
│   ├── 1_Dashboard.py        # Analytics dashboard
│   ├── 2_Predict.py          # Individual prediction
│   ├── 3_Batch_Upload.py     # Batch CSV upload & predict
│   ├── 4_Model_Training.py   # Model training & comparison
│   └── 5_History.py          # Prediction history
├── datasets/                 # Existing datasets
└── assets/                   # Static assets (logo, etc.)
```

## Datasets Analysis
1. **Student_Performance(1).csv** (10,000 rows) - Hours Studied, Previous Scores, Extracurricular, Sleep Hours, Sample Question Papers, Performance Index
2. **StudentsPerformance.csv** (1,000 rows) - Gender, Race, Parental Education, Lunch, Test Prep, Math/Reading/Writing scores
3. **student-mat.csv** (395 rows) - School, Demographics, Family, Grades G1/G2/G3

**Primary dataset**: [student-mat.csv](file:///d:/Personal%20Projects/Mini-Project_3rd%20Year/datasets/student/student-mat.csv) (most aligned with project goals - has attendance, grades, study habits)

## Features to Build
1. ✅ Data preprocessing pipeline
2. ✅ ML model training (3 algorithms comparison)
3. ✅ Risk level classification (Low/Medium/High)
4. ✅ Academic Risk Advisor Agent (rule-based recommendations)
5. ✅ Streamlit dashboard with charts
6. ✅ Individual student prediction form
7. ✅ Batch CSV upload
8. ✅ Prediction history storage (Supabase)
9. ✅ Model performance metrics display

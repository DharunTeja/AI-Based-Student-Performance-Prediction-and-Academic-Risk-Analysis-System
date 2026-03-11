"""
config.py - Configuration settings for the Student Performance Prediction System.
Contains all constants, database credentials, and application settings.
"""

import os
from dotenv import load_dotenv  # Load environment variables from .env file

# Load environment variables from a .env file (for sensitive credentials)
load_dotenv()

# ============================================================
# SUPABASE DATABASE CONFIGURATION
# ============================================================
# Supabase provides a free PostgreSQL database with REST API
# You need to create a project at https://supabase.com and get these values
SUPABASE_URL = os.getenv("SUPABASE_URL", "")  # Your Supabase project URL
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")  # Your Supabase anon/public key

# ============================================================
# DATASET PATHS
# ============================================================
# Path to the primary dataset used for training the ML model
# student-mat.csv has the most relevant features: attendance, grades, demographics
PRIMARY_DATASET = os.path.join("datasets", "student", "student-mat.csv")

# Secondary datasets for additional analysis
SECONDARY_DATASET_1 = os.path.join("datasets", "Student_Performance(1).csv")
SECONDARY_DATASET_2 = os.path.join("datasets", "StudentsPerformance.csv")

# ============================================================
# MODEL CONFIGURATION
# ============================================================
# Directory where trained ML models (.pkl files) will be saved
MODEL_SAVE_DIR = os.path.join("models", "saved_models")

# Test-train split ratio: 80% training, 20% testing
TEST_SIZE = 0.2

# Random seed for reproducibility (ensures same results every run)
RANDOM_STATE = 42

# ============================================================
# RISK LEVEL THRESHOLDS
# ============================================================
# These thresholds determine how students are classified into risk categories
# Based on the probability of passing (predicted by the ML model)
HIGH_RISK_THRESHOLD = 0.4    # P(pass) < 0.4 → High Risk
MEDIUM_RISK_THRESHOLD = 0.7  # 0.4 ≤ P(pass) < 0.7 → Medium Risk
# P(pass) ≥ 0.7 → Low Risk (no threshold needed, it's the default)

# ============================================================
# APPLICATION SETTINGS
# ============================================================
APP_TITLE = "🎓 AI Student Performance Predictor"
APP_ICON = "🎓"
APP_LAYOUT = "wide"  # Use wide layout for better dashboard visualization

# Color scheme for risk levels (used in charts and badges)
RISK_COLORS = {
    "High Risk": "#FF4B4B",    # Red - immediate attention needed
    "Medium Risk": "#FFA500",  # Orange - needs monitoring
    "Low Risk": "#00CC66",     # Green - performing well
}

# Grade mapping for the student-mat dataset
# G3 is the final grade (0-20 scale), we map to Pass/Fail
PASS_THRESHOLD = 10  # Students scoring >= 10 (out of 20) are considered passing

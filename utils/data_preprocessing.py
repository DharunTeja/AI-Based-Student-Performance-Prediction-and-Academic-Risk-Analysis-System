"""
data_preprocessing.py - Data Cleaning & Feature Engineering Module.

This module handles all data preprocessing tasks:
1. Loading raw CSV datasets
2. Handling missing values
3. Encoding categorical features (converting text to numbers)
4. Normalizing numerical features (scaling to 0-1 range)
5. Feature selection (picking the most relevant columns)
6. Splitting data into training and testing sets

The primary dataset used is student-mat.csv which contains:
- Demographics: school, sex, age, address, family size
- Academic: study time, failures, grades (G1, G2, G3)
- Social: going out, alcohol consumption, health
- Support: extra educational support, family support
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding & scaling
from sklearn.model_selection import train_test_split  # For splitting data
import os
import config  # Our configuration file with paths and constants


def load_primary_dataset():
    """
    Load the primary dataset (student-mat.csv).
    
    This dataset contains 395 students with 33 attributes.
    The semicolon (;) is used as the delimiter in this CSV file.
    
    Returns:
        pd.DataFrame: The loaded dataset, or None if file not found
    """
    try:
        # student-mat.csv uses semicolon as separator (not comma)
        df = pd.read_csv(config.PRIMARY_DATASET, delimiter=";")
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at: {config.PRIMARY_DATASET}")
        return None


def load_secondary_dataset_1():
    """
    Load the Student_Performance(1).csv dataset.
    
    This dataset has 10,000 records with features like:
    Hours Studied, Previous Scores, Extracurricular Activities,
    Sleep Hours, Sample Question Papers Practiced, Performance Index
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        df = pd.read_csv(config.SECONDARY_DATASET_1)
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at: {config.SECONDARY_DATASET_1}")
        return None


def load_secondary_dataset_2():
    """
    Load the StudentsPerformance.csv dataset.
    
    This dataset has 1,000 records with features like:
    Gender, Race/Ethnicity, Parental Education, 
    Math Score, Reading Score, Writing Score
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    try:
        df = pd.read_csv(config.SECONDARY_DATASET_2)
        return df
    except FileNotFoundError:
        print(f"❌ Dataset not found at: {config.SECONDARY_DATASET_2}")
        return None


def preprocess_primary_dataset(df):
    """
    Clean and preprocess the primary dataset (student-mat.csv) for ML training.
    
    Steps performed:
    1. Create a binary target variable (Pass/Fail) from G3 grade
    2. Encode all categorical columns (text → numbers)
    3. Select relevant features for prediction
    4. Handle any missing values
    
    Args:
        df (pd.DataFrame): Raw dataframe loaded from student-mat.csv
    
    Returns:
        tuple: (X_features, y_target, feature_names, label_encoders)
            - X_features: DataFrame of input features
            - y_target: Series of target values (0=Fail, 1=Pass)
            - feature_names: List of feature column names
            - label_encoders: Dict of LabelEncoder objects (needed for decoding later)
    """
    # Step 1: Create binary target variable from final grade (G3)
    # G3 ranges from 0-20; students scoring >= 10 are considered "Pass"
    df["pass_fail"] = (df["G3"] >= config.PASS_THRESHOLD).astype(int)  # 1 = Pass, 0 = Fail
    
    # Step 2: Create a risk score (continuous) for more nuanced analysis
    # Normalize G3 to 0-1 range (0 = worst, 1 = best)
    df["performance_score"] = df["G3"] / 20.0
    
    # Step 3: Identify categorical columns (columns with text/string values)
    # These need to be converted to numbers before ML algorithms can use them
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    
    # Step 4: Encode categorical variables using LabelEncoder
    # LabelEncoder converts each unique string to a unique integer
    # Example: "GP" → 0, "MS" → 1
    label_encoders = {}  # Store encoders so we can decode predictions later
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Transform text to numbers
        label_encoders[col] = le  # Save the encoder for this column
    
    # Step 5: Select features for prediction
    # We exclude G3 (final grade) because that's what we're predicting
    # We also exclude our derived columns (pass_fail, performance_score)
    features_to_exclude = ["G3", "pass_fail", "performance_score"]
    feature_names = [col for col in df.columns if col not in features_to_exclude]
    
    # Step 6: Extract features (X) and target (y)
    X = df[feature_names]  # All columns except the target
    y = df["pass_fail"]    # Binary target: 0 = Fail, 1 = Pass
    
    # Step 7: Handle missing values by filling with column median
    # Median is preferred over mean because it's less affected by outliers
    X = X.fillna(X.median())
    
    return X, y, feature_names, label_encoders


def get_risk_level(probability):
    """
    Classify a student into a risk category based on their predicted 
    probability of passing.
    
    Risk Levels:
    - High Risk:   P(pass) < 0.4  → Student is very likely to fail
    - Medium Risk: 0.4 ≤ P(pass) < 0.7 → Student needs attention
    - Low Risk:    P(pass) ≥ 0.7  → Student is on track
    
    Args:
        probability (float): Predicted probability of passing (0.0 to 1.0)
    
    Returns:
        str: Risk level category ("High Risk", "Medium Risk", or "Low Risk")
    """
    if probability < config.HIGH_RISK_THRESHOLD:
        return "High Risk"      # Urgent attention needed
    elif probability < config.MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"    # Needs monitoring
    else:
        return "Low Risk"       # Performing well


def prepare_training_data(X, y):
    """
    Split the dataset into training and testing sets.
    
    We use 80% of data for training the model and 20% for testing.
    The random_state ensures we get the same split every time (reproducibility).
    stratify=y ensures both sets have the same ratio of Pass/Fail students.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - Split datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,       # 20% for testing
        random_state=config.RANDOM_STATE,  # For reproducibility
        stratify=y  # Maintain class balance in both sets
    )
    return X_train, X_test, y_train, y_test


def get_feature_descriptions():
    """
    Return human-readable descriptions for each feature in the primary dataset.
    This helps faculty understand what each input field means.
    
    Returns:
        dict: Mapping of column names to their descriptions
    """
    return {
        "school": "Student's school (GP = Gabriel Pereira, MS = Mousinho da Silveira)",
        "sex": "Student's sex (F = Female, M = Male)",
        "age": "Student's age (from 15 to 22)",
        "address": "Home address type (U = Urban, R = Rural)",
        "famsize": "Family size (LE3 = ≤ 3, GT3 = > 3)",
        "Pstatus": "Parent's cohabitation status (T = Together, A = Apart)",
        "Medu": "Mother's education (0=None, 1=Primary, 2=5th-9th, 3=Secondary, 4=Higher)",
        "Fedu": "Father's education (0=None, 1=Primary, 2=5th-9th, 3=Secondary, 4=Higher)",
        "Mjob": "Mother's job (teacher, health, services, at_home, other)",
        "Fjob": "Father's job (teacher, health, services, at_home, other)",
        "reason": "Reason to choose this school (home, reputation, course, other)",
        "guardian": "Student's guardian (mother, father, other)",
        "traveltime": "Home to school travel time (1=<15min, 2=15-30min, 3=30-60min, 4=>60min)",
        "studytime": "Weekly study time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)",
        "failures": "Number of past class failures (0 to 4)",
        "schoolsup": "Extra educational support (yes/no)",
        "famsup": "Family educational support (yes/no)",
        "paid": "Extra paid classes (yes/no)",
        "activities": "Extra-curricular activities (yes/no)",
        "nursery": "Attended nursery school (yes/no)",
        "higher": "Wants to take higher education (yes/no)",
        "internet": "Internet access at home (yes/no)",
        "romantic": "In a romantic relationship (yes/no)",
        "famrel": "Quality of family relationships (1=Very Bad to 5=Excellent)",
        "freetime": "Free time after school (1=Very Low to 5=Very High)",
        "goout": "Going out with friends (1=Very Low to 5=Very High)",
        "Dalc": "Workday alcohol consumption (1=Very Low to 5=Very High)",
        "Walc": "Weekend alcohol consumption (1=Very Low to 5=Very High)",
        "health": "Current health status (1=Very Bad to 5=Very Good)",
        "absences": "Number of school absences (0 to 93)",
        "G1": "First period grade (0 to 20)",
        "G2": "Second period grade (0 to 20)",
    }

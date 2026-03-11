"""
predictor.py - Prediction Logic Module.

This module provides the interface for making predictions using a trained model.
It handles:
1. Loading the saved model
2. Preprocessing new student input data
3. Making predictions (Pass/Fail)
4. Calculating risk levels
5. Generating recommendations via the Risk Advisor Agent

This is the module that connects the frontend to the ML model.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_preprocessing import get_risk_level
from utils.risk_advisor import generate_recommendations


def load_model():
    """
    Load the trained ML model bundle from disk.
    
    The bundle contains:
    - model: The trained sklearn classifier
    - scaler: The StandardScaler fitted on training data
    - feature_names: List of expected input feature names
    - label_encoders: Encoders for categorical features
    
    Returns:
        dict or None: The model bundle, or None if not found
    """
    model_path = os.path.join(config.MODEL_SAVE_DIR, "best_model.pkl")
    
    if os.path.exists(model_path):
        try:
            bundle = joblib.load(model_path)  # Load the .pkl file
            return bundle
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"⚠️ No trained model found at {model_path}")
        print("   Please train a model first using the Model Training page.")
        return None


def predict_single_student(student_data, model_bundle=None):
    """
    Make a prediction for a single student.
    
    This function takes a dictionary of student attributes, processes it
    through the ML model, and returns the prediction with risk level
    and recommendations.
    
    Workflow:
    1. Load model if not provided
    2. Create a DataFrame from the input (model expects DataFrame format)
    3. Encode categorical features (same encoding as training)
    4. Scale features (same scaling as training)
    5. Predict using the model
    6. Calculate risk level from probability
    7. Generate recommendations from the Risk Advisor Agent
    
    Args:
        student_data (dict): Dictionary with student attribute values
            Example: {"age": 17, "sex": "M", "studytime": 2, ...}
        model_bundle (dict, optional): Pre-loaded model bundle.
            If None, loads from disk.
    
    Returns:
        dict: Prediction result containing:
            - 'prediction': 0 (Fail) or 1 (Pass)
            - 'prediction_label': "Pass" or "Fail"
            - 'pass_probability': float (0.0 to 1.0)
            - 'risk_level': "High Risk", "Medium Risk", or "Low Risk"
            - 'recommendations': List of recommendation dicts
            - 'risk_summary': Summary of recommendations
        Returns None if prediction fails.
    """
    # Step 1: Load model if not provided
    if model_bundle is None:
        model_bundle = load_model()
        if model_bundle is None:
            return None
    
    model = model_bundle["model"]           # The ML classifier
    scaler = model_bundle["scaler"]         # For scaling features
    feature_names = model_bundle["feature_names"]  # Expected columns
    label_encoders = model_bundle["label_encoders"]  # For encoding text
    
    # Step 2: Create a DataFrame from the input dictionary
    # The model was trained on a DataFrame, so we need the same format
    input_df = pd.DataFrame([student_data])
    
    # Step 3: Ensure all required features are present
    # If a feature is missing, fill with 0 (default value)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Step 4: Encode categorical features using the same encoders from training
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                # Try to encode the value using the stored encoder
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                # If the value wasn't seen during training, use 0
                input_df[col] = 0
    
    # Step 5: Select only the features the model was trained on, in the right order
    input_features = input_df[feature_names]
    
    # Step 6: Scale features using the same scaler from training
    input_scaled = scaler.transform(input_features)
    
    # Step 7: Make the prediction
    prediction = model.predict(input_scaled)[0]  # 0 = Fail, 1 = Pass
    
    # Step 8: Get prediction probabilities
    # predict_proba returns [[P(Fail), P(Pass)]]
    probabilities = model.predict_proba(input_scaled)[0]
    pass_probability = probabilities[1]  # Probability of passing (index 1)
    
    # Step 9: Determine risk level based on pass probability
    risk_level = get_risk_level(pass_probability)
    
    # Step 10: Generate recommendations from the Risk Advisor Agent
    recommendations = generate_recommendations(
        student_data=student_data,
        risk_level=risk_level,
        pass_probability=pass_probability
    )
    
    # Step 11: Compile and return the result
    result = {
        "prediction": int(prediction),
        "prediction_label": "Pass" if prediction == 1 else "Fail",
        "pass_probability": float(pass_probability),
        "fail_probability": float(probabilities[0]),
        "risk_level": risk_level,
        "recommendations": recommendations,
        "risk_summary": {
            "total": len(recommendations),
            "critical": sum(1 for r in recommendations if r["priority"] == "Critical"),
            "important": sum(1 for r in recommendations if r["priority"] == "Important"),
            "suggested": sum(1 for r in recommendations if r["priority"] == "Suggested"),
        }
    }
    
    return result


def predict_batch(students_df, model_bundle=None):
    """
    Make predictions for a batch of students from a CSV/DataFrame.
    
    This is used when faculty uploads a CSV file with multiple students.
    
    Args:
        students_df (pd.DataFrame): DataFrame with student data
        model_bundle (dict, optional): Pre-loaded model bundle
    
    Returns:
        pd.DataFrame: Original data with added prediction columns:
            - prediction: Pass/Fail
            - pass_probability: Probability of passing
            - risk_level: High/Medium/Low Risk
    """
    if model_bundle is None:
        model_bundle = load_model()
        if model_bundle is None:
            return None
    
    # Process each student individually and collect results
    results = []
    for idx, row in students_df.iterrows():
        student_data = row.to_dict()  # Convert each row to a dictionary
        result = predict_single_student(student_data, model_bundle)
        
        if result:
            results.append({
                "prediction": result["prediction_label"],
                "pass_probability": round(result["pass_probability"], 3),
                "risk_level": result["risk_level"],
            })
        else:
            results.append({
                "prediction": "Error",
                "pass_probability": 0,
                "risk_level": "Unknown",
            })
    
    # Add prediction columns to the original DataFrame
    results_df = pd.DataFrame(results)
    output_df = pd.concat([students_df.reset_index(drop=True), results_df], axis=1)
    
    return output_df

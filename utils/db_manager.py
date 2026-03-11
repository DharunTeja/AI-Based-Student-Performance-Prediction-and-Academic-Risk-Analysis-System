"""
db_manager.py - Database Operations Module (Supabase Integration).

This module handles all database operations using Supabase (hosted PostgreSQL).
If Supabase credentials are not configured, it falls back to local JSON storage.

Features:
- Save prediction history
- Retrieve past predictions
- Store model performance metrics
- Faculty authentication (simplified)

Supabase provides:
- Free PostgreSQL database
- REST API for CRUD operations
- Real-time subscriptions
- Authentication built-in
"""

import json
import os
from datetime import datetime

# Try to import supabase; if not installed, use local fallback
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

import config

# =================================================================
# LOCAL JSON FALLBACK STORAGE
# =================================================================
# When Supabase is not configured, we store data in local JSON files
# This ensures the app works even without database setup
LOCAL_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
PREDICTIONS_FILE = os.path.join(LOCAL_STORAGE_DIR, "predictions_history.json")
MODELS_FILE = os.path.join(LOCAL_STORAGE_DIR, "model_metrics.json")


def _ensure_local_storage():
    """Create local storage directory and files if they don't exist."""
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)  # Create directory
    
    # Create predictions file if it doesn't exist
    if not os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump([], f)  # Empty list of predictions
    
    # Create models file if it doesn't exist
    if not os.path.exists(MODELS_FILE):
        with open(MODELS_FILE, "w") as f:
            json.dump([], f)  # Empty list of model metrics


def get_supabase_client():
    """
    Create and return a Supabase client connection.
    
    Requires SUPABASE_URL and SUPABASE_KEY to be set in config.py or .env file.
    
    Returns:
        Client or None: Supabase client if configured, None otherwise
    """
    if SUPABASE_AVAILABLE and config.SUPABASE_URL and config.SUPABASE_KEY:
        try:
            client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            return client
        except Exception as e:
            print(f"⚠️ Supabase connection failed: {e}")
            return None
    return None


def save_prediction(student_data, prediction, risk_level, probability, recommendations):
    """
    Save a prediction record to the database (or local storage).
    
    Each prediction record contains:
    - Student input data
    - Prediction result (Pass/Fail)
    - Risk level classification
    - Pass probability
    - Generated recommendations
    - Timestamp
    
    Args:
        student_data (dict): The student's input features
        prediction (int): 0 = Fail, 1 = Pass
        risk_level (str): "High Risk", "Medium Risk", or "Low Risk"
        probability (float): Probability of passing
        recommendations (list): List of recommendation dicts
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Create the prediction record
    record = {
        "student_data": student_data,
        "prediction": "Pass" if prediction == 1 else "Fail",
        "risk_level": risk_level,
        "pass_probability": round(probability, 4),
        "recommendations_count": len(recommendations),
        "timestamp": datetime.now().isoformat(),  # Current date and time
    }
    
    # Try Supabase first, fall back to local storage
    client = get_supabase_client()
    
    if client:
        try:
            # Insert into Supabase "predictions" table
            # The table must be created in Supabase dashboard first
            client.table("predictions").insert({
                "student_data": json.dumps(student_data),
                "prediction": record["prediction"],
                "risk_level": risk_level,
                "pass_probability": record["pass_probability"],
                "recommendations_count": record["recommendations_count"],
                "created_at": record["timestamp"],
            }).execute()
            return True
        except Exception as e:
            print(f"⚠️ Supabase save failed, using local storage: {e}")
    
    # Fallback: Save to local JSON file
    _ensure_local_storage()
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)  # Load existing predictions
        
        predictions.append(record)  # Add new prediction
        
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump(predictions, f, indent=2, default=str)  # Save back
        
        return True
    except Exception as e:
        print(f"❌ Failed to save prediction: {e}")
        return False


def get_prediction_history(limit=50):
    """
    Retrieve prediction history from the database (or local storage).
    
    Returns the most recent predictions, sorted by timestamp (newest first).
    
    Args:
        limit (int): Maximum number of records to return (default: 50)
    
    Returns:
        list: List of prediction records
    """
    # Try Supabase first
    client = get_supabase_client()
    
    if client:
        try:
            response = client.table("predictions") \
                .select("*") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            return response.data
        except Exception as e:
            print(f"⚠️ Supabase fetch failed, using local storage: {e}")
    
    # Fallback: Read from local JSON file
    _ensure_local_storage()
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            predictions = json.load(f)
        
        # Sort by timestamp (newest first) and limit results
        predictions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return predictions[:limit]
    except Exception as e:
        print(f"❌ Failed to read prediction history: {e}")
        return []


def save_model_metrics(model_name, metrics):
    """
    Save ML model performance metrics for comparison.
    
    Stores accuracy, precision, recall, F1-score for each trained model.
    
    Args:
        model_name (str): Name of the ML algorithm (e.g., "Random Forest")
        metrics (dict): Performance metrics dictionary
    
    Returns:
        bool: True if saved successfully
    """
    record = {
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Try Supabase first
    client = get_supabase_client()
    if client:
        try:
            client.table("model_metrics").insert({
                "model_name": model_name,
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "created_at": record["timestamp"],
            }).execute()
            return True
        except Exception as e:
            print(f"⚠️ Supabase save failed: {e}")
    
    # Fallback to local storage
    _ensure_local_storage()
    try:
        with open(MODELS_FILE, "r") as f:
            models = json.load(f)
        models.append(record)
        with open(MODELS_FILE, "w") as f:
            json.dump(models, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"❌ Failed to save model metrics: {e}")
        return False


def get_model_metrics():
    """
    Retrieve all saved model performance metrics.
    
    Returns:
        list: List of model metric records
    """
    client = get_supabase_client()
    if client:
        try:
            response = client.table("model_metrics") \
                .select("*") \
                .order("created_at", desc=True) \
                .execute()
            return response.data
        except Exception:
            pass
    
    # Fallback
    _ensure_local_storage()
    try:
        with open(MODELS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def clear_prediction_history():
    """
    Clear all prediction history.
    Useful for resetting the system or during testing.
    
    Returns:
        bool: True if cleared successfully
    """
    _ensure_local_storage()
    try:
        with open(PREDICTIONS_FILE, "w") as f:
            json.dump([], f)
        return True
    except Exception:
        return False

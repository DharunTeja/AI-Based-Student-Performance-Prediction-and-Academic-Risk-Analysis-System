"""
train_model.py - Machine Learning Model Training Pipeline.

This module handles the complete ML training workflow:
1. Load and preprocess the dataset
2. Train 3 different classification algorithms
3. Evaluate each model's performance
4. Compare and select the best model
5. Save the best model to disk (.pkl file)

Algorithms used:
- Logistic Regression: Simple linear classifier, good baseline
- Decision Tree: Rule-based classifier, easy to interpret
- Random Forest: Ensemble of decision trees, usually most accurate

The trained model is saved using joblib (efficient for sklearn models).
"""

import os
import sys
import joblib  # For saving/loading ML models efficiently
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression    # Algorithm 1
from sklearn.tree import DecisionTreeClassifier         # Algorithm 2
from sklearn.ensemble import RandomForestClassifier     # Algorithm 3
from sklearn.metrics import (
    accuracy_score,       # Overall correct predictions / total predictions
    precision_score,      # True positives / (True positives + False positives)
    recall_score,         # True positives / (True positives + False negatives)
    f1_score,             # Harmonic mean of precision and recall
    classification_report,  # Detailed per-class metrics
    confusion_matrix,     # Matrix showing TP, TN, FP, FN counts
)
from sklearn.preprocessing import StandardScaler  # Feature scaling

# Add parent directory to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_preprocessing import (
    load_primary_dataset,
    preprocess_primary_dataset,
    prepare_training_data,
)


def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all three ML models and evaluate their performance.
    
    This function:
    1. Scales the features (standardization to mean=0, std=1)
    2. Trains Logistic Regression, Decision Tree, and Random Forest
    3. Evaluates each model on the test set
    4. Returns all results for comparison
    
    Why we scale features:
    - Logistic Regression is sensitive to feature scales
    - Features like 'absences' (0-93) and 'studytime' (1-4) have very different ranges
    - Scaling puts all features on the same scale for fair comparison
    
    Args:
        X_train: Training features
        y_train: Training labels (0=Fail, 1=Pass)
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Results for each model containing:
            - 'model': The trained model object
            - 'scaler': The fitted StandardScaler
            - 'metrics': Dict of accuracy, precision, recall, f1
            - 'predictions': Predictions on test set
            - 'probabilities': Prediction probabilities
            - 'confusion_matrix': Confusion matrix
    """
    # Step 1: Scale features using StandardScaler
    # StandardScaler transforms each feature: (value - mean) / std_deviation
    # This makes all features have mean=0 and std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data, then transform
    X_test_scaled = scaler.transform(X_test)         # Only transform test data (no fit!)
    # Note: We only fit on training data to prevent "data leakage"
    # If we fit on test data too, the model would have "seen" test data distribution
    
    # Step 2: Define the three models to train
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,           # Maximum iterations for convergence
            random_state=config.RANDOM_STATE,  # For reproducibility
            solver='lbfgs',          # Optimization algorithm (good for small datasets)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,            # Limit tree depth to prevent overfitting
            random_state=config.RANDOM_STATE,
            min_samples_split=5,     # Minimum samples to split a node
            min_samples_leaf=2,      # Minimum samples in a leaf node
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,        # Number of trees in the forest
            max_depth=15,            # Maximum depth per tree
            random_state=config.RANDOM_STATE,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,               # Use all CPU cores for parallel training
        ),
    }
    
    results = {}  # Store results for each model
    
    # Step 3: Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")
        
        # Train the model on scaled training data
        model.fit(X_train_scaled, y_train)
        
        # Make predictions on test data
        predictions = model.predict(X_test_scaled)
        
        # Get prediction probabilities (needed for risk level classification)
        # predict_proba returns [P(Fail), P(Pass)] for each student
        probabilities = model.predict_proba(X_test_scaled)
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)
        
        # Print results
        print(f"Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nDetailed Report:\n{classification_report(y_test, predictions, zero_division=0)}")
        
        # Store results
        results[name] = {
            "model": model,
            "scaler": scaler,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            },
            "predictions": predictions,
            "probabilities": probabilities,
            "confusion_matrix": cm,
        }
    
    return results


def select_best_model(results):
    """
    Select the best performing model based on accuracy.
    
    We use accuracy as the primary metric because:
    - It's easy to understand (% of correct predictions)
    - Our dataset is reasonably balanced
    - For imbalanced datasets, F1 score would be better
    
    Args:
        results (dict): Results from train_all_models()
    
    Returns:
        tuple: (best_model_name, best_model_results)
    """
    best_name = None
    best_accuracy = 0
    
    for name, result in results.items():
        acc = result["metrics"]["accuracy"]
        if acc > best_accuracy:
            best_accuracy = acc
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_accuracy:.1%})")
    return best_name, results[best_name]


def save_model(model, scaler, feature_names, label_encoders, model_name="best_model"):
    """
    Save the trained model and associated objects to disk.
    
    We save:
    1. The trained ML model
    2. The StandardScaler (needed to scale new input data the same way)
    3. Feature names (to know which columns the model expects)
    4. Label encoders (to convert categorical inputs from text to numbers)
    
    All objects are saved in a single .pkl file for convenience.
    
    Args:
        model: Trained sklearn model object
        scaler: Fitted StandardScaler object
        feature_names: List of feature column names
        label_encoders: Dict of LabelEncoder objects
        model_name: Name for the saved file (default: "best_model")
    
    Returns:
        str: Path where the model was saved
    """
    # Create the save directory if it doesn't exist
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    # Bundle everything into a single dictionary
    model_bundle = {
        "model": model,              # The trained ML model
        "scaler": scaler,            # For scaling new inputs
        "feature_names": feature_names,  # Expected input columns
        "label_encoders": label_encoders,  # For encoding categorical inputs
    }
    
    # Save using joblib (more efficient than pickle for sklearn models)
    save_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.pkl")
    joblib.dump(model_bundle, save_path)
    
    print(f"✅ Model saved to: {save_path}")
    return save_path


def load_saved_model(model_name="best_model"):
    """
    Load a previously saved model from disk.
    
    Args:
        model_name: Name of the saved model file (without .pkl extension)
    
    Returns:
        dict or None: Model bundle containing model, scaler, feature_names, 
                      label_encoders. Returns None if file not found.
    """
    load_path = os.path.join(config.MODEL_SAVE_DIR, f"{model_name}.pkl")
    
    if os.path.exists(load_path):
        model_bundle = joblib.load(load_path)
        print(f"✅ Model loaded from: {load_path}")
        return model_bundle
    else:
        print(f"❌ No saved model found at: {load_path}")
        return None


def run_training_pipeline():
    """
    Execute the complete training pipeline from start to finish.
    
    This is the main function that orchestrates the entire training process:
    1. Load dataset
    2. Preprocess data
    3. Split into train/test
    4. Train all models
    5. Select best model
    6. Save best model
    
    Returns:
        tuple: (results_dict, best_model_name, feature_names) or None if failed
    """
    print("🔄 Starting Training Pipeline...")
    print("=" * 60)
    
    # Step 1: Load the dataset
    print("\n📁 Step 1: Loading dataset...")
    df = load_primary_dataset()
    if df is None:
        print("❌ Failed to load dataset. Aborting.")
        return None
    print(f"   Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Step 2: Preprocess the data
    print("\n🔧 Step 2: Preprocessing data...")
    X, y, feature_names, label_encoders = preprocess_primary_dataset(df)
    print(f"   Features: {len(feature_names)} columns")
    print(f"   Pass: {y.sum()} students, Fail: {len(y) - y.sum()} students")
    
    # Step 3: Split into training and testing sets
    print("\n✂️ Step 3: Splitting dataset...")
    X_train, X_test, y_train, y_test = prepare_training_data(X, y)
    print(f"   Training set: {len(X_train)} records")
    print(f"   Testing set:  {len(X_test)} records")
    
    # Step 4: Train all models
    print("\n🤖 Step 4: Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Select the best model
    print("\n🏆 Step 5: Selecting best model...")
    best_name, best_result = select_best_model(results)
    
    # Step 6: Save the best model
    print("\n💾 Step 6: Saving best model...")
    save_model(
        model=best_result["model"],
        scaler=best_result["scaler"],
        feature_names=feature_names,
        label_encoders=label_encoders,
        model_name="best_model"
    )
    
    print("\n" + "=" * 60)
    print("✅ Training Pipeline Complete!")
    print(f"🏆 Best Model: {best_name}")
    print(f"📊 Accuracy: {best_result['metrics']['accuracy']:.1%}")
    print("=" * 60)
    
    return results, best_name, feature_names


# If this file is run directly (not imported), execute the training pipeline
if __name__ == "__main__":
    run_training_pipeline()

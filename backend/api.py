"""
api.py - Flask REST API for EduInsight AI.

This module wraps the existing Python ML logic (models, utils, config)
into RESTful API endpoints for the React frontend to consume.

Run with:
    python api.py
"""

import os
import sys
import json
import io
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add project root to path so we can import existing modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root so config paths resolve correctly
os.chdir(PROJECT_ROOT)

import config
from utils.data_preprocessing import (
    load_primary_dataset,
    load_secondary_dataset_1,
    load_secondary_dataset_2,
    preprocess_primary_dataset,
    prepare_training_data,
    get_risk_level,
    get_feature_descriptions,
)
from models.predictor import load_model, predict_single_student, predict_batch
from models.train_model import train_all_models, select_best_model, save_model
from utils.db_manager import (
    save_prediction,
    get_prediction_history,
    save_model_metrics,
    get_model_metrics,
    clear_prediction_history,
)

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React dev server


# ============================================================
# CUSTOM JSON ENCODER (handles numpy types)
# ============================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


app.json_encoder = NumpyEncoder


# ============================================================
# HELPER: Convert DataFrame to JSON-safe dict
# ============================================================
def df_to_json(df):
    """Convert DataFrame to a list of dicts, handling numpy types."""
    return json.loads(df.to_json(orient="records"))


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route("/", methods=["GET"])
def index():
    """Root endpoint to verify the API is running."""
    return jsonify({
        "message": "🎓 EduInsight AI API is running successfully!",
        "status": "online",
        "docs": "Endpoints available under /api/"
    })

@app.route("/api/status", methods=["GET"])
def get_status():
    """Check system status: model trained? dataset available?"""
    model_exists = os.path.exists(
        os.path.join(config.MODEL_SAVE_DIR, "best_model.pkl")
    )
    dataset_exists = os.path.exists(config.PRIMARY_DATASET)

    return jsonify({
        "model_trained": model_exists,
        "dataset_loaded": dataset_exists,
        "risk_advisor_active": True,
    })


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Get quick statistics from the primary dataset."""
    df = load_primary_dataset()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    total = len(df)
    pass_count = int((df["G3"] >= config.PASS_THRESHOLD).sum())
    fail_count = int((df["G3"] < config.PASS_THRESHOLD).sum())
    avg_grade = round(float(df["G3"].mean()), 1)
    median_grade = float(df["G3"].median())

    return jsonify({
        "total_students": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_percentage": round(pass_count / total * 100, 1),
        "fail_percentage": round(fail_count / total * 100, 1),
        "avg_grade": avg_grade,
        "median_grade": median_grade,
    })


@app.route("/api/feature-descriptions", methods=["GET"])
def get_features():
    """Get human-readable feature descriptions."""
    return jsonify(get_feature_descriptions())


@app.route("/api/dashboard/<dataset>", methods=["GET"])
def get_dashboard_data(dataset):
    """Get dashboard chart data for a specific dataset."""

    if dataset == "primary":
        df = load_primary_dataset()
        if df is None:
            return jsonify({"error": "Primary dataset not found"}), 404

        df["pass_fail"] = df["G3"].apply(
            lambda x: "Pass" if x >= config.PASS_THRESHOLD else "Fail"
        )
        df["risk_level"] = df["G3"].apply(lambda x: get_risk_level(x / 20.0))

        # Overview metrics
        total = len(df)
        avg_grade = round(float(df["G3"].mean()), 1)
        pass_rate = round(float((df["pass_fail"] == "Pass").mean() * 100), 1)
        avg_absences = round(float(df["absences"].mean()), 1)
        num_features = len(df.columns)

        # Grade distribution (G3 values with pass_fail)
        grade_distribution = df_to_json(df[["G3", "pass_fail"]])

        # Pass/Fail counts
        pass_fail_counts = df["pass_fail"].value_counts().to_dict()

        # Risk level counts
        risk_counts = df["risk_level"].value_counts().to_dict()

        # Grade progression (G1, G2, G3 values)
        grade_progression = {
            "G1": df["G1"].tolist(),
            "G2": df["G2"].tolist(),
            "G3": df["G3"].tolist(),
        }

        # Absences vs G3
        absences_vs_grade = df_to_json(df[["absences", "G3", "pass_fail"]])

        # Study time vs grade
        study_labels = {1: "< 2 hrs", 2: "2-5 hrs", 3: "5-10 hrs", 4: "> 10 hrs"}
        study_avg = df.groupby("studytime")["G3"].mean().reset_index()
        study_avg["label"] = study_avg["studytime"].map(study_labels)
        study_avg_data = df_to_json(study_avg)

        # Feature correlations (numeric only)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()
        corr_data = {
            "columns": numeric_cols,
            "values": corr_matrix.values.tolist(),
        }

        # Top correlations with G3
        g3_corr = corr_matrix["G3"].drop("G3").sort_values(ascending=False)
        top_positive = {k: round(float(v), 3) for k, v in g3_corr.head(5).items()}
        top_negative = {k: round(float(v), 3) for k, v in g3_corr.tail(5).items()}

        return jsonify({
            "metrics": {
                "total": total,
                "avg_grade": avg_grade,
                "pass_rate": pass_rate,
                "avg_absences": avg_absences,
                "num_features": num_features,
            },
            "grade_distribution": grade_distribution,
            "pass_fail_counts": pass_fail_counts,
            "risk_counts": risk_counts,
            "grade_progression": grade_progression,
            "absences_vs_grade": absences_vs_grade,
            "study_avg": study_avg_data,
            "correlation": corr_data,
            "top_positive_corr": top_positive,
            "top_negative_corr": top_negative,
            "pass_threshold": config.PASS_THRESHOLD,
        })

    elif dataset == "performance":
        df = load_secondary_dataset_1()
        if df is None:
            return jsonify({"error": "Performance Index dataset not found"}), 404

        df = df.dropna(subset=["Performance Index"])

        return jsonify({
            "metrics": {
                "total": len(df),
                "avg_performance": round(float(df["Performance Index"].mean()), 1),
                "avg_hours": round(float(df["Hours Studied"].mean()), 1),
                "avg_sleep": round(float(df["Sleep Hours"].mean()), 1),
            },
            "data": df_to_json(df),
        })

    elif dataset == "exams":
        df = load_secondary_dataset_2()
        if df is None:
            return jsonify({"error": "Exam Scores dataset not found"}), 404

        return jsonify({
            "metrics": {
                "total": len(df),
                "avg_math": round(float(df["math score"].mean()), 1),
                "avg_reading": round(float(df["reading score"].mean()), 1),
                "avg_writing": round(float(df["writing score"].mean()), 1),
            },
            "data": df_to_json(df),
        })

    return jsonify({"error": "Unknown dataset"}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict performance for a single student."""
    student_data = request.json

    if not student_data:
        return jsonify({"error": "No student data provided"}), 400

    model_bundle = load_model()
    if model_bundle is None:
        return jsonify({"error": "Model not trained. Train the model first."}), 400

    result = predict_single_student(student_data, model_bundle)

    if result is None:
        return jsonify({"error": "Prediction failed"}), 500

    # Save to history
    save_prediction(
        student_data=student_data,
        prediction=result["prediction"],
        risk_level=result["risk_level"],
        probability=result["pass_probability"],
        recommendations=result["recommendations"],
    )

    return jsonify(result)


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """Predict performance for multiple students from a CSV file."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        content = file.read().decode("utf-8")
        # Try semicolon first, then comma
        df_uploaded = pd.read_csv(io.StringIO(content), delimiter=";")
        if len(df_uploaded.columns) <= 2:
            df_uploaded = pd.read_csv(io.StringIO(content), delimiter=",")
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    model_bundle = load_model()
    if model_bundle is None:
        return jsonify({"error": "Model not trained"}), 400

    results_df = predict_batch(df_uploaded, model_bundle)

    if results_df is None:
        return jsonify({"error": "Batch prediction failed"}), 500

    # Save each prediction to history
    saved_count = 0
    for _, row in results_df.iterrows():
        student_data = row.drop(
            ["Prediction", "Pass Probability", "Risk Level"]
        ).to_dict()
        save_prediction(
            student_data=student_data,
            prediction=1 if row["Prediction"] == "Pass" else 0,
            risk_level=row["Risk Level"],
            probability=row["Pass Probability"],
            recommendations=[],
        )
        saved_count += 1

    # Summary stats
    total = len(results_df)
    pass_count = int((results_df["Prediction"] == "Pass").sum())
    fail_count = int((results_df["Prediction"] == "Fail").sum())
    high_risk = int((results_df["Risk Level"] == "High Risk").sum())

    return jsonify({
        "total": total,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "high_risk": high_risk,
        "results": df_to_json(results_df),
        "saved_count": saved_count,
        "predictions": results_df["Prediction"].value_counts().to_dict(),
        "risk_levels": results_df["Risk Level"].value_counts().to_dict(),
        "probabilities": results_df["Pass Probability"].tolist(),
    })


@app.route("/api/train", methods=["POST"])
def train_models():
    """Train all ML models and return comparison results."""
    df = load_primary_dataset()
    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    # Preprocess
    X, y, feature_names, label_encoders = preprocess_primary_dataset(df.copy())
    X_train, X_test, y_train, y_test = prepare_training_data(X, y)

    # Train
    results = train_all_models(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_result = select_best_model(results)

    # Save best model
    save_model(
        model=best_result["model"],
        scaler=best_result["scaler"],
        feature_names=feature_names,
        label_encoders=label_encoders,
        model_name="best_model",
    )

    # Save metrics for each model
    for name, result in results.items():
        save_model_metrics(name, result["metrics"])

    # Build response
    comparison = []
    for name, result in results.items():
        metrics = result["metrics"]
        cm = result["confusion_matrix"].tolist()
        entry = {
            "name": name,
            "metrics": metrics,
            "confusion_matrix": cm,
            "is_best": name == best_name,
        }
        comparison.append(entry)

    # Feature importance (Random Forest)
    feature_importance = []
    if "Random Forest" in results:
        rf_model = results["Random Forest"]["model"]
        importances = rf_model.feature_importances_
        for feat, imp in zip(feature_names, importances):
            feature_importance.append({
                "feature": feat,
                "importance": round(float(imp), 4),
            })
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    return jsonify({
        "best_model": best_name,
        "best_accuracy": best_result["metrics"]["accuracy"],
        "comparison": comparison,
        "feature_importance": feature_importance,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "num_features": len(feature_names),
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Get prediction history."""
    limit = request.args.get("limit", 100, type=int)
    history = get_prediction_history(limit=limit)
    return jsonify(history)


@app.route("/api/history/clear", methods=["DELETE"])
def clear_history():
    """Clear all prediction history."""
    success = clear_prediction_history()
    if success:
        return jsonify({"message": "History cleared successfully"})
    return jsonify({"error": "Failed to clear history"}), 500


@app.route("/api/model-metrics", methods=["GET"])
def get_metrics():
    """Get saved model metrics."""
    metrics = get_model_metrics()
    return jsonify(metrics)


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting EduInsight AI API Server...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"🌐 API available at: http://localhost:5000")
    app.run(debug=True, port=5000, host="0.0.0.0")

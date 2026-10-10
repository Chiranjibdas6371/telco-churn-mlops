import os
import time
from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/telco_churn.joblib"))
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
app = Flask(__name__)

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
PREDICTIONS = Counter("api_prediction_total", "Predictions by class", ["prediction"])
ERRORS = Counter("api_prediction_errors_total", "Prediction errors")
DRIFT = Gauge("data_drift_score", "Latest feature drift score")

REQUIRED = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]

@app.get("/health")
def health():
    REQUESTS.labels("health", "GET", "200").inc()
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.post("/predict")
def predict():
    start = time.perf_counter()
    status = "200"
    try:
        payload = request.get_json(force=True)
        missing = [x for x in REQUIRED if x not in payload]
        if missing:
            status = "400"
            return jsonify({"error": "missing fields", "fields": missing}), 400
        if model is None:
            status = "503"
            return jsonify({"error": "model not loaded"}), 503
        X = pd.DataFrame([payload])[REQUIRED]
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])
        PREDICTIONS.labels(str(prediction)).inc()
        return jsonify({"churn": prediction, "churn_probability": round(probability, 5)})
    except Exception as exc:
        ERRORS.inc()
        status = "500"
        return jsonify({"error": str(exc)}), 500
    finally:
        REQUESTS.labels("predict", "POST", status).inc()
        LATENCY.labels("predict").observe(time.perf_counter() - start)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

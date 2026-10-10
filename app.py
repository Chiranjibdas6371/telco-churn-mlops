
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST


app = Flask(__name__)

# Load the trained pipeline
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/best_model.pkl")
)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


# Prometheus metrics
REQUESTS = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status"]
)

LATENCY = Histogram(
    "api_request_latency_seconds",
    "API request latency",
    ["endpoint"]
)

PREDICTIONS = Counter(
    "api_prediction_total",
    "Predictions by class",
    ["prediction"]
)

ERRORS = Counter(
    "api_prediction_errors_total",
    "Prediction errors"
)

DRIFT = Gauge(
    "data_drift_score",
    "Latest feature drift score"
)


@app.before_request
def before_request_metrics():
    request._metrics_start_time = time.perf_counter()


@app.after_request
def after_request_metrics(response):
    endpoint = request.endpoint or "unknown"
    method = request.method
    status = str(response.status_code)

    REQUESTS.labels(
        endpoint=endpoint,
        method=method,
        status=status
    ).inc()

    start_time = getattr(request, "_metrics_start_time", None)
    if start_time is not None and endpoint != "metrics":
        LATENCY.labels(endpoint=endpoint).observe(
            time.perf_counter() - start_time
        )

    return response


# Original features expected from the website/API
REQUIRED = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


def prepare_features(payload):
    """Validate input and reproduce training feature engineering."""

    missing = [column for column in REQUIRED if column not in payload]

    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    data = {column: payload[column] for column in REQUIRED}

    # Convert numerical inputs
    for column in [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]:
        try:
            data[column] = float(data[column])
        except (ValueError, TypeError):
            raise ValueError(f"{column} must be numeric")

    if data["SeniorCitizen"] not in (0, 1):
        raise ValueError("SeniorCitizen must be 0 or 1")

    if data["tenure"] < 0 or data["tenure"] > 1000:
        raise ValueError("Invalid tenure value")

    if data["MonthlyCharges"] < 0 or data["TotalCharges"] < 0:
        raise ValueError("Charges cannot be negative")

    df = pd.DataFrame([data])

    # Same ServiceCount logic used during training
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["ServiceCount"] = sum(
        (df[column] == "Yes").astype(int)
        for column in service_columns
    )

    # Same tenure bins and labels used during training
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["New", "Early", "Established", "Loyal"]
    ).astype(str)

    return df


def predict_customer(payload):
    """Return the churn prediction and probability."""

    if model is None:
        raise RuntimeError("Model is not loaded")

    X = prepare_features(payload)

    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    PREDICTIONS.labels(str(prediction)).inc()

    return {
        "churn": prediction,
        "prediction_label": "Likely to Churn" if prediction == 1
                            else "Likely to Stay",
        "churn_probability": round(probability, 5)
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    status = "healthy" if model is not None else "model_not_loaded"

    return jsonify({
        "status": status,
        "model_loaded": model is not None
    }), 200 if model is not None else 503


@app.route("/metrics")
def metrics():
    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


@app.route("/predict", methods=["POST"])
def predict():
    start = time.perf_counter()
    status = "200"

    try:
        if request.is_json:
            payload = request.get_json(silent=True)

            if not isinstance(payload, dict):
                status = "400"
                return jsonify({"error": "A JSON object is required"}), 400

            result = predict_customer(payload)
            return jsonify(result)

        # HTML form submission
        payload = request.form.to_dict()
        result = predict_customer(payload)

        return render_template("result.html", result=result)

    except ValueError as exc:
        status = "400"

        if request.is_json:
            return jsonify({"error": str(exc)}), 400

        return render_template(
            "result.html",
            error=str(exc),
            result=None
        ), 400

    except RuntimeError:
        status = "503"

        if request.is_json:
            return jsonify({"error": "Model is not loaded"}), 503

        return render_template(
            "result.html",
            error="The prediction model is unavailable.",
            result=None
        ), 503

    except Exception:
        ERRORS.inc()
        status = "500"

        if request.is_json:
            return jsonify({"error": "Prediction failed"}), 500

        return render_template(
            "result.html",
            error="Something went wrong. Please try again.",
            result=None
        ), 500

    finally:
        REQUESTS.labels("predict", request.method, status).inc()
        LATENCY.labels("predict").observe(
            time.perf_counter() - start
        )


if __name__ == "__main__":
    app.run(port=5000)

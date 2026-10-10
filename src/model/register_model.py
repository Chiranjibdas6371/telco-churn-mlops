import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
MLFLOW_TRACKING_URI = "http://16.4.53.144:5000"
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn-random-forest"

def register_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    model_path = "models/best_model.pkl"
    model=joblib.load(model_path)
    with open("reports/evaluation_metrics.json","r") as file:
        metrics=json.load(file)
    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    input_example = X_train.head(5)
    predictions = model.predict(input_example)
    signature=infer_signature(input_example,predictions)
    with mlflow.start_run(run_name="Random Forest model registration"):
        mlflow.log_params({
            "model_type": "RandomForestClassifier",
            "model_source": model_path
        })
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=["numpy.dtype"]
        )

        print("Model logged and registered successfully.")
        print(f"Registered model: {REGISTERED_MODEL_NAME}")
        print(f"Model URI: {model_info.model_uri}")

if __name__ =="__main__":
    register_model()
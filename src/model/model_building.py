import os
import joblib
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_params():
    with open("params.yaml","r") as file:
        params=yaml.safe_load(file)
    return params

def load_data():
    data_path="data/processed/train.csv"
    df=pd.read_csv(data_path)
    X_train=df.drop(columns=["Churn", "customerID"], errors="ignore")
    y_train = df["Churn"]
    return X_train,y_train
def build_pipeline(params):
    numerical_features = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "ServiceCount"
    ]

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
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
        "TenureGroup"
    ]

    numerical_transformer=Pipeline(
        [("imputer",SimpleImputer(strategy="mean")),
         ("scaler",StandardScaler())]
    )
    categorical_transformer= Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)
    model_params = params["model_building"]

    rf_model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        min_samples_split=model_params["min_samples_split"],
        min_samples_leaf=model_params["min_samples_leaf"],
        max_features=model_params["max_features"],
        class_weight=model_params["class_weight"],
        random_state=42,
        n_jobs=-1
    )
    pipeline=Pipeline(
        [
            ("preprocessor",preprocessor),
            ("model",rf_model)
        ]
    )
    return pipeline
def train_model():
    params = load_params()
    X_train, y_train = load_data()
    if y_train.isna().any():
        raise ValueError("Unexpected Churn labels. Expected 'Yes' or 'No'.")
    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    model_path = "models/best_model.pkl"
    joblib.dump(pipeline, model_path)
    print("Model training completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_model()



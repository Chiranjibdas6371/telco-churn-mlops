
import os
import json
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,classification_report,confusion_matrix

def load_model():
    model_path = "models/best_model.pkl"
    model = joblib.load(model_path)
    return model


def load_test_data():
    test_path = "data/processed/test.csv"
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["Churn", "customerID"],errors="ignore")

    y_test = test_df["Churn"]

    if y_test.isna().any():
        raise ValueError(
            "Unexpected Churn labels. Expected 'Yes' or 'No'."
        )
    return X_test, y_test


def evaluate_model():
    model = load_model()
    X_test, y_test = load_test_data()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_prob)
        )
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs("reports", exist_ok=True)

    with open("reports/evaluation_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    with open("reports/classification_report.json", "w") as file:
        json.dump(report, file, indent=4)

    with open("reports/confusion_matrix.json", "w") as file:
        json.dump(cm, file, indent=4)

    print("\nModel Evaluation Results")
    print("-" * 35)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(cm)

    print("\nEvaluation results saved in reports/")


if __name__ == "__main__":
    evaluate_model()

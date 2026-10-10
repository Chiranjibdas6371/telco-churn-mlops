import pandas as pd
import os


def data_ingestion():

    input_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    output_path = "data/interim/telco_churn.csv"

    # Read data
    df = pd.read_csv(input_path)

    print("Data loaded successfully")
    print("Shape:", df.shape)

    # Expected columns
    expected_columns = [
        "customerID",
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
        "TotalCharges",
        "Churn"
    ]

    # Column validation
    if list(df.columns) != expected_columns:
        print("Column validation failed!")
        print("Expected columns:")
        print(expected_columns)
        print("Actual columns:")
        print(list(df.columns))
        return

    print("Column validation successful")

    # Create interim folder
    os.makedirs("data/interim", exist_ok=True)

    # Save data
    df.to_csv(output_path, index=False)

    print("Data saved to:", output_path)


if __name__ == "__main__":
    data_ingestion()
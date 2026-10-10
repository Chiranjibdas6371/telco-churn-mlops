import os
import pandas as pd

from sklearn.model_selection import train_test_split


INTERIM_DATA_PATH = "data/interim/telco_churn.csv"

PROCESSED_DATA_DIR = "data/processed"

TRAIN_DATA_PATH = "data/processed/train.csv"

TEST_DATA_PATH = "data/processed/test.csv"


def load_data():

    if not os.path.exists(INTERIM_DATA_PATH):
        raise FileNotFoundError(
            "Dataset not found: " + INTERIM_DATA_PATH
        )

    df = pd.read_csv(INTERIM_DATA_PATH)

    print("Dataset shape:", df.shape)

    return df


def clean_data(df):

    df = df.copy()

    # --------------------------------
    # Remove customer ID
    # --------------------------------

    df.drop(
        columns=["customerID"],
        inplace=True
    )

    # --------------------------------
    # Convert TotalCharges to numeric
    # --------------------------------

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Handle missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # --------------------------------
    # Feature Engineering
    # --------------------------------

    # Tenure Group

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, float("inf")],
        labels=[
            "New",
            "Early",
            "Established",
            "Loyal"
        ]
    )

    # Service Count

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

    df["ServiceCount"] = (
        df[service_columns] == "Yes"
    ).sum(axis=1)

    # --------------------------------
    # Convert target variable
    # --------------------------------

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df


def split_data(df):

    # Features
    X = df.drop(
        columns=["Churn"]
    )

    # Target
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Combine features and target

    train_data = X_train.copy()
    train_data["Churn"] = y_train

    test_data = X_test.copy()
    test_data["Churn"] = y_test

    return train_data, test_data


def save_data(train_data, test_data):

    os.makedirs(
        PROCESSED_DATA_DIR,
        exist_ok=True
    )

    train_data.to_csv(
        TRAIN_DATA_PATH,
        index=False
    )

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    print("\nProcessed data saved successfully.")

    print(
        "Training data:",
        TRAIN_DATA_PATH
    )

    print(
        "Testing data:",
        TEST_DATA_PATH
    )


def process_data():

    # Load data
    df = load_data()

    # Clean data + Feature Engineering
    df = clean_data(df)

    print(
        "Shape after cleaning:",
        df.shape
    )

    # Show engineered features
    print("\nEngineered features:")
    print(
        df[
            [
                "TenureGroup",
                "ServiceCount"
            ]
        ].head()
    )

    # Split data
    train_data, test_data = split_data(df)

    print(
        "\nTraining data shape:",
        train_data.shape
    )

    print(
        "Testing data shape:",
        test_data.shape
    )

    # Save data
    save_data(
        train_data,
        test_data
    )


if __name__ == "__main__":

    process_data()
from pathlib import Path
import hashlib

import joblib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# 1. PROJECT PATHS AND CONFIGURATION

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"
CLEANED_DATA_PATH = ROOT / "data" / "processed" / "train_cleaned.csv"
ORIGINAL_DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
MLFLOW_DB = ROOT / "mlflow.db"
EXPERIMENT_NAME = "MP219_Canteen_Prediction"
MODEL_NAME = "MP219_Canteen_Model"


# 2. DATA HASHING

def calculate_data_hash(file_path):
    """Calculate SHA256 hash of the original training dataset."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# 3. LOAD CONFIGURATION


def load_params():
    """Load project parameters from params.yaml."""

    with open(PARAMS_PATH, "r") as file:
        params = yaml.safe_load(file)

    return params


# 4. LOAD CLEANED DATA

def load_data():
    """Load the cleaned training dataset."""

    df = pd.read_csv(CLEANED_DATA_PATH)

    print(f"Cleaned dataset shape: {df.shape}")

    return df


# 5. PREPROCESSING


def create_preprocessor():
    """Create numerical and categorical preprocessing pipelines."""

    numeric_features = [
        "students_on_campus",
        "menu_price_rs",
        "temp_c",
        "exam_days_left",
        "queue_minutes_yesterday",
    ]

    categorical_features = [
        "special_dishes",
        "day_type",
        "block",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor


# 6. TRAIN FINAL MODEL

def train():

    # Load data
    df = load_data()

    # Separate features and target
    X = df.drop(
        columns=["record_id", "date", "plates_sold"]
    )

    y = df["plates_sold"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Create preprocessing pipeline
    preprocessor = create_preprocessor()

    # Final selected model: Linear Regression
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    # Train
    print("\nTraining Linear Regression...")

    pipeline.fit(X_train, y_train)

    # Evaluate
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    print(f"\nLinear Regression MAE: {mae:.3f}")

    # Calculate original dataset hash
    dataset_hash = calculate_data_hash(ORIGINAL_DATA_PATH)


    # 7. MLFLOW TRACKING

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="linear_regression_final"):

        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("selected_from", "model_experiments_notebook")
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("features_count", X.shape[1])
        mlflow.log_param("data_sha256", dataset_hash)

        # Log metric
        mlflow.log_metric("mae", mae)

        # Log model and register it
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
            input_example=X_train.head(3),
        )

        run_id = mlflow.active_run().info.run_id

        print(f"\nMLflow Run ID: {run_id}")
        print(f"Model URI: {model_info.model_uri}")


    # 8. SET CHAMPION ALIAS

    client = mlflow.tracking.MlflowClient()

    # Get the latest registered version
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    latest_version = max(
        versions,
        key=lambda version: int(version.version)
    )

    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        latest_version.version,
    )

    print(
        f"\nChampion model: {MODEL_NAME}"
        f" | Version: {latest_version.version}"
    )

    # 9. SAVE MODEL LOCALLY

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved locally at: {MODEL_PATH}")

    return {
        "mae": mae,
        "run_id": run_id,
        "model_version": latest_version.version,
        "model_path": MODEL_PATH,
    }


# 10. EXECUTION

if __name__ == "__main__":

    results = train()

    print("\n========== TRAINING SUMMARY ==========")
    print(f"Model: Linear Regression")
    print(f"MAE: {results['mae']:.3f}")
    print(f"MLflow Run ID: {results['run_id']}")
    print(f"Model Version: {results['model_version']}")
    print(f"Saved Model: {results['model_path']}")
    print("===================================")
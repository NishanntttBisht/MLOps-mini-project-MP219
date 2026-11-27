from pathlib import Path
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from canteen.data_cleaning import clean_training_data
from canteen.model_training import create_preprocessor

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/raw/train.csv"
LATER_PATH = ROOT / "data/raw/later.csv"
MODEL_PATH = ROOT / "models/model.joblib"
FINDINGS_PATH = ROOT / "reports/findings.yaml"
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_NAME = "MP219_Canteen_Model"
EXPERIMENT_NAME = "MP219_Canteen_Prediction"

FEATURES_TO_DROP = ["record_id", "date", "plates_sold"]


def main():
    print("Loading data...")

    train = clean_training_data(pd.read_csv(TRAIN_PATH))
    later = clean_training_data(pd.read_csv(LATER_PATH))

    later = later.sort_values("date").reset_index(drop=True)

    split = int(len(later) * 0.70)

    later_train = later.iloc[:split]
    later_validation = later.iloc[split:]

    print(f"New data for retraining: {len(later_train)}")
    print(f"Validation data: {len(later_validation)}")

    # Combine original training data with new 70%
    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True
    )

    X_train = combined_train.drop(columns=FEATURES_TO_DROP)
    y_train = combined_train["plates_sold"]

    X_val = later_validation.drop(columns=FEATURES_TO_DROP)
    y_val = later_validation["plates_sold"]

    # Load existing v1 champion
    v1 = joblib.load(MODEL_PATH)

    v1_predictions = v1.predict(X_val)
    v1_mae = mean_absolute_error(y_val, v1_predictions)

    print(f"\nV1 MAE on validation data: {v1_mae:.4f}")

    # Train v2
    v2 = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LinearRegression()),
    ])

    print("\nTraining V2...")
    v2.fit(X_train, y_train)

    v2_predictions = v2.predict(X_val)
    v2_mae = mean_absolute_error(y_val, v2_predictions)

    improvement = ((v1_mae - v2_mae) / v1_mae) * 100

    print(f"V2 MAE: {v2_mae:.4f}")
    print(f"Improvement: {improvement:.2f}%")

    promoted = improvement >= 8

    print("\nPROMOTION DECISION:", "PROMOTED" if promoted else "NOT PROMOTED")

    # MLflow tracking
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="retraining_v2") as run:
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("retraining_data_percent", 70)
        mlflow.log_param("validation_data_percent", 30)
        mlflow.log_param("promotion_threshold_percent", 8)

        mlflow.log_metric("v1_validation_mae", v1_mae)
        mlflow.log_metric("v2_validation_mae", v2_mae)
        mlflow.log_metric("improvement_percent", improvement)

        mlflow.log_param("promoted", promoted)

        if promoted:
            model_info = mlflow.sklearn.log_model(
                sk_model=v2,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                serialization_format="cloudpickle",
                input_example=X_train.head(3),
            )

            print(f"V2 registered: {model_info.model_uri}")

    # If V2 promoted, update local champion model
    if promoted:
        joblib.dump(v2, MODEL_PATH)

        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}'"
        )

        latest_version = max(
            versions,
            key=lambda v: int(v.version)
        )

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            latest_version.version
        )

        print(f"New champion version: {latest_version.version}")
    else:
        print("V1 remains champion.")

    # Update official findings
    import yaml

    with open(FINDINGS_PATH) as f:
        findings = yaml.safe_load(f)

    findings["monitoring"]["mae_v1_on_check"] = round(float(v1_mae), 4)
    findings["monitoring"]["mae_v2_on_check"] = round(float(v2_mae), 4)
    findings["monitoring"]["decision"] = (
        "promoted" if promoted else "not_promoted"
    )

    findings["tracking"]["experiment_name"] = EXPERIMENT_NAME
    findings["tracking"]["registered_model_name"] = MODEL_NAME

    if promoted:
        findings["tracking"]["champion_version"] = str(latest_version.version)
    else:
        findings["tracking"]["champion_version"] = "1"

    with open(FINDINGS_PATH, "w") as f:
        yaml.safe_dump(findings, f, sort_keys=False)

    print("\n========== RETRAINING SUMMARY ==========")
    print(f"V1 MAE: {v1_mae:.4f}")
    print(f"V2 MAE: {v2_mae:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Decision: {'PROMOTED' if promoted else 'NOT PROMOTED'}")
    print("====================================")


if __name__ == "__main__":
    main()
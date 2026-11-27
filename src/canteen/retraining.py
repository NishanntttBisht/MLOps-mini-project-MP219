from pathlib import Path
import hashlib
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from canteen.data_cleaning import clean_training_data

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/raw/train.csv"
LATER_PATH = ROOT / "data/raw/later.csv"
MODEL_PATH = ROOT / "models/model.joblib"
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_NAME = "MP219_Canteen_Model"
EXPERIMENT_NAME = "MP219_Canteen_Prediction"

PROMOTION_THRESHOLD = 0.08
NEW_DATA_RATIO = 0.70


def create_preprocessor():
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
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    return preprocessor


def create_model():
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", LinearRegression()),
    ])


def main():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading datasets...")

    # Only the original training dataset gets training corrections
    train_raw = pd.read_csv(TRAIN_PATH)
    train = clean_training_data(train_raw)

    # Later data is already clean — do NOT apply training fixes
    later = pd.read_csv(LATER_PATH)

    later = later.sort_values("date").reset_index(drop=True)

    split_index = int(len(later) * NEW_DATA_RATIO)

    new_training_data = later.iloc[:split_index].copy()
    validation_data = later.iloc[split_index:].copy()

    print(f"Cleaned original training data: {len(train)}")
    print(f"New data for retraining: {len(new_training_data)}")
    print(f"Validation data: {len(validation_data)}")

    feature_columns = [
        "students_on_campus",
        "menu_price_rs",
        "temp_c",
        "exam_days_left",
        "special_dishes",
        "queue_minutes_yesterday",
        "day_type",
        "block",
    ]

    X_val = validation_data[feature_columns]
    y_val = validation_data["plates_sold"]

    # Load original V1 explicitly from MLflow
    v1 = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}/1"
    )

    v1_predictions = v1.predict(X_val)
    v1_mae = mean_absolute_error(y_val, v1_predictions)

    print(f"\nV1 validation MAE: {v1_mae:.4f}")

    # Combine cleaned original train + untouched new data
    combined_train = pd.concat(
        [train, new_training_data],
        ignore_index=True
    )

    X_train = combined_train[feature_columns]
    y_train = combined_train["plates_sold"]

    v2 = create_model()
    v2.fit(X_train, y_train)

    v2_predictions = v2.predict(X_val)
    v2_mae = mean_absolute_error(y_val, v2_predictions)

    improvement = (v1_mae - v2_mae) / v1_mae

    promoted = improvement >= PROMOTION_THRESHOLD

    print(f"V2 validation MAE: {v2_mae:.4f}")
    print(f"Improvement: {improvement:.2%}")
    print(f"Promotion decision: {'PROMOTED' if promoted else 'REJECTED'}")

    # Dataset fingerprint for reproducibility
    data_hash = hashlib.sha256(
        train_raw.to_csv(index=False).encode()
    ).hexdigest()

    with mlflow.start_run(run_name="retraining_v2") as run:

        mlflow.log_param("original_train_rows", len(train))
        mlflow.log_param("new_data_rows", len(new_training_data))
        mlflow.log_param("validation_rows", len(validation_data))
        mlflow.log_param("new_data_ratio", NEW_DATA_RATIO)
        mlflow.log_param("promotion_threshold", PROMOTION_THRESHOLD)
        mlflow.log_param("data_sha256", data_hash)

        mlflow.log_metric("v1_validation_mae", v1_mae)
        mlflow.log_metric("v2_validation_mae", v2_mae)
        mlflow.log_metric("improvement_pct", improvement * 100)

        mlflow.log_param("promotion_decision", promoted)

        run_id = run.info.run_id

        if promoted:

            print("\nRegistering V2 model...")

            model_info = mlflow.sklearn.log_model(
                sk_model=v2,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=["numpy.dtype"],
            )

            client = mlflow.tracking.MlflowClient()

            # Fetch the newly created version
            versions = client.search_model_versions(
                f"name='{MODEL_NAME}'"
            )

            new_version = max(
                int(version.version)
                for version in versions
            )

            client.set_registered_model_alias(
                MODEL_NAME,
                "champion",
                str(new_version),
            )

            champion = v2
            champion_version = new_version

            print(f"V2 promoted as version {champion_version}")

        else:

            print("\nV2 did not meet promotion threshold.")
            print("Keeping V1 as champion.")

            client = mlflow.tracking.MlflowClient()

            client.set_registered_model_alias(
                MODEL_NAME,
                "champion",
                "1",
            )

            champion = v1
            champion_version = 1

    # Save champion locally for API / Docker / Kubernetes
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, MODEL_PATH)

    print(f"\nChampion model version: {champion_version}")
    print(f"Saved champion to: {MODEL_PATH}")

    # Update findings.yaml
    findings_path = ROOT / "reports" / "findings.yaml"

    with open(findings_path, "r") as file:
        findings = yaml.safe_load(file)

    findings["monitoring"]["mae_v1_on_check"] = round(
        float(v1_mae), 4
    )
    findings["monitoring"]["mae_v2_on_check"] = round(
        float(v2_mae), 4
    )
    findings["monitoring"]["decision"] = (
        "promoted_v2" if promoted else "kept_v1"
    )

    findings["tracking"]["experiment_name"] = EXPERIMENT_NAME
    findings["tracking"]["best_run_id"] = run_id
    findings["tracking"]["registered_model_name"] = MODEL_NAME
    findings["tracking"]["champion_version"] = int(champion_version)

    with open(findings_path, "w") as file:
        yaml.safe_dump(findings, file, sort_keys=False)

    print("\nRetraining workflow completed successfully.")
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_absolute_error
from canteen.data_cleaning import clean_training_data

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
OUTPUT_PATH = ROOT / "reports" / "monitoring_results.json"
MLFLOW_DB = ROOT / "mlflow.db"

MODEL_NAME = "MP219_Canteen_Model"

NUMERIC_COLUMNS = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "special_dishes",
    "queue_minutes_yesterday",
    "plates_sold",
]


def calculate_psi(reference, current, bins=10):
    """Population Stability Index: how far `current` has drifted from `reference`."""
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if reference.empty or current.empty:
        return None

    # Bin edges are quantiles of the reference data
    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    # Too few distinct values to form meaningful bins
    if len(edges) < 3:
        return 0.0

    # Open-ended outer bins so out-of-range values are still counted
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    # Clip to avoid log(0) and division by zero in empty bins
    ref_pct = np.clip(ref_pct, 0.0001, None)
    cur_pct = np.clip(cur_pct, 0.0001, None)

    return float(
        np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    )


def main():
    print("Loading data...")

    train_raw = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)  # later data is already clean

    train = clean_training_data(train_raw)

    # Use the registered V1 model explicitly, not the local V2 champion
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

    v1_uri = f"models:/{MODEL_NAME}/1"
    print(f"Loading original V1 from: {v1_uri}")

    v1 = mlflow.sklearn.load_model(v1_uri)

    # PSI for all numeric columns (training vs later data)
    print("\nCalculating PSI...")

    psi_results = {}

    for column in NUMERIC_COLUMNS:
        psi_results[column] = calculate_psi(
            train[column],
            later[column],
            bins=10
        )

    # Column with the highest PSI; None values are treated as lowest
    most_drifted = max(
        psi_results,
        key=lambda column: (
            psi_results[column]
            if psi_results[column] is not None
            else -1
        )
    )

    print("\nPSI RESULTS")
    for column, value in psi_results.items():
        print(f"{column}: {value:.4f}" if value is not None else f"{column}: N/A")

    print(f"\nMost drifted column: {most_drifted}")

    # V1 MAE on clean later data by block
    print("\nCalculating V1 MAE by block...")

    # Drop identifiers and the target before predicting
    X_later = later.drop(columns=["record_id", "date", "plates_sold"])
    y_later = later["plates_sold"]

    predictions = v1.predict(X_later)

    later_eval = later.copy()
    later_eval["prediction"] = predictions
    later_eval["absolute_error"] = abs(
        later_eval["plates_sold"] - later_eval["prediction"]
    )

    mae_by_group = (
        later_eval.groupby("block")["absolute_error"]
        .mean()
        .to_dict()
    )

    # Block with the highest error is the worst served
    worst_group = max(mae_by_group, key=mae_by_group.get)

    print("\nV1 MAE BY BLOCK")
    for block, mae in mae_by_group.items():
        print(f"{block}: {mae:.4f}")

    print(f"\nWorst-served group: {worst_group}")

    results = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
    }

    # Save results as JSON, creating the reports folder if needed
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
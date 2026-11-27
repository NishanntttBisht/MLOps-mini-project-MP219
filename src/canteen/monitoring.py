from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error
from canteen.data_cleaning import clean_training_data

ROOT = Path(__file__).resolve().parents[2]

NUMERIC_FEATURES = [
    "students_on_campus",
    "menu_price_rs",
    "temp_c",
    "exam_days_left",
    "queue_minutes_yesterday",
]

MODEL_PATH = ROOT / "models" / "model.joblib"
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
OUTPUT_PATH = ROOT / "reports" / "monitoring_results.json"


def calculate_psi(reference, current, bins=10):
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    if len(reference) == 0 or len(current) == 0:
        return None

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    ref_pct = np.clip(ref_pct, 0.0001, None)
    cur_pct = np.clip(cur_pct, 0.0001, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def main():
    print("Loading datasets...")

    train_raw = pd.read_csv(TRAIN_PATH)
    later_raw = pd.read_csv(LATER_PATH)

    train = clean_training_data(train_raw)
    later = clean_training_data(later_raw)

    model = joblib.load(MODEL_PATH)

    print("Calculating PSI...")

    psi_results = {}

    for feature in NUMERIC_FEATURES:
        psi_results[feature] = calculate_psi(
            train[feature], later[feature]
        )

    most_drifted = max(
        psi_results,
        key=lambda x: psi_results[x] if psi_results[x] is not None else -1
    )

    print("\nPSI RESULTS")
    for feature, value in psi_results.items():
        print(f"{feature}: {value:.4f}" if value is not None else f"{feature}: N/A")

    print(f"\nMost drifted feature: {most_drifted}")

    print("\nCalculating MAE by block...")

    features = later.drop(
        columns=["record_id", "date", "plates_sold"]
    )
    actual = later["plates_sold"]

    predictions = model.predict(features)

    later["prediction"] = predictions
    later["absolute_error"] = abs(actual - predictions)

    mae_by_group = (
        later.groupby("block")["absolute_error"]
        .mean()
        .to_dict()
    )

    worst_group = max(mae_by_group, key=mae_by_group.get)

    print("\nMAE BY BLOCK")
    for block, mae in mae_by_group.items():
        print(f"{block}: {mae:.3f}")

    print(f"\nWorst-served group: {worst_group}")

    results = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
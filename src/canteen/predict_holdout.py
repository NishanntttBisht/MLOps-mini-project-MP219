from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

HOLDOUT_PATH = ROOT / "data/raw/holdout_features.csv"
MODEL_PATH = ROOT / "models/model.joblib"
OUTPUT_PATH = ROOT / "predictions/holdout_predictions.csv"


def main():
    print("Loading clean holdout dataset...")

    holdout = pd.read_csv(HOLDOUT_PATH)

    print(f"Holdout rows: {len(holdout)}")

    model = joblib.load(MODEL_PATH)

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

    X_holdout = holdout[feature_columns]

    print("Generating predictions...")
    predictions = model.predict(X_holdout)

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nPrediction summary:")
    print(output["prediction"].describe())

    print(f"\nSaved predictions: {OUTPUT_PATH}")
    print(f"Total predictions: {len(output)}")
    print(f"Unique record IDs: {output['record_id'].nunique()}")
    print(f"Missing predictions: {output['prediction'].isna().sum()}")


if __name__ == "__main__":
    main()
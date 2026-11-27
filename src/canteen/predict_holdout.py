from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "model.joblib"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
OUTPUT_PATH = ROOT / "predictions" / "holdout_predictions.csv"

def clean_holdout(df):
    df = df.copy()

    # Correct B-Block queue time unit mix-up
    mask = df["block"] == "B-Block"
    df.loc[mask, "queue_minutes_yesterday"] /= 60

    # Replace temperature sentinel
    df["temp_c"] = df["temp_c"].replace(999, float("nan"))

    # Normalize day type spellings
    df["day_type"] = df["day_type"].str.strip().str.lower()
    df["day_type"] = df["day_type"].replace({
        "wee.": "weekday",
        "sat.": "saturday",
        "hol.": "holiday",
    })

    return df


def main():
    print("Loading holdout data and champion model...")

    holdout = pd.read_csv(HOLDOUT_PATH)
    model = joblib.load(MODEL_PATH)

    cleaned = clean_holdout(holdout)

    X = cleaned.drop(columns=["record_id", "date"])

    predictions = model.predict(X)
    predictions = [max(0, round(float(p))) for p in predictions]

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nHoldout prediction summary")
    print("Rows predicted:", len(output))
    print("Mean prediction:", round(output["prediction"].mean(), 2))
    print("Minimum prediction:", output["prediction"].min())
    print("Maximum prediction:", output["prediction"].max())
    print("\nOutput saved to:", OUTPUT_PATH)
    print("\nPreview:")
    print(output.head())


if __name__ == "__main__":
    main()
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PARAMS_PATH = ROOT / "params.yaml"

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

settings = params["cleaning"]


def clean_training_data(df):
    """
    Apply corrections ONLY to the messy original training dataset.
    Do not use this function on later.csv or holdout_features.csv.
    """
    df = df.copy()

    # 1. Remove duplicate rows
    if settings["remove_duplicates"]:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Correct B-Block queue-time unit mix-up
    mask = df["block"] == settings["unit_mixup_group"]

    df.loc[mask, "queue_minutes_yesterday"] = (
        df.loc[mask, "queue_minutes_yesterday"]
        / settings["unit_conversion_factor"]
    )

    # 3. Replace invalid temperature sentinel with missing values
    df["temp_c"] = df["temp_c"].replace(
        settings["temperature_sentinel"],
        float("nan")
    )

    # 4. Correct target typos
    mask = df["plates_sold"] > settings["target_typo_threshold"]

    df.loc[mask, "plates_sold"] = (
        df.loc[mask, "plates_sold"]
        / settings["target_correction_factor"]
    )

    # 5. Standardize day_type category spellings
    df["day_type"] = df["day_type"].str.strip().str.lower()

    df["day_type"] = df["day_type"].replace(
        settings["day_type_mapping"]
    )

    return df


def load_training_data():
    """Load and clean the original messy training dataset."""
    train_path = params["data"]["train_path"]

    df = pd.read_csv(train_path)

    return clean_training_data(df)


def load_later_data():
    """
    Load later.csv as-is.
    This dataset is already clean, so no training corrections are applied.
    """
    later_path = ROOT / "data" / "raw" / "later.csv"

    return pd.read_csv(later_path)


def load_holdout_data():
    """
    Load September holdout features as-is.
    No training-specific corrections are applied.
    """
    holdout_path = ROOT / "data" / "raw" / "holdout_features.csv"

    return pd.read_csv(holdout_path)


if __name__ == "__main__":
    train_raw = pd.read_csv(params["data"]["train_path"])
    train_cleaned = clean_training_data(train_raw)

    print("Training dataset")
    print("Original shape:", train_raw.shape)
    print("Cleaned shape:", train_cleaned.shape)
    print("Remaining duplicates:", train_cleaned.duplicated().sum())

    print("\nQueue statistics:")
    print(
        train_cleaned.groupby("block")[
            "queue_minutes_yesterday"
        ].describe()
    )

    print("\nTemperature:")
    print("Missing:", train_cleaned["temp_c"].isna().sum())

    print("\nTarget:")
    print("Max plates sold:", train_cleaned["plates_sold"].max())

    print("\nDay types:")
    print(train_cleaned["day_type"].unique())

    cleaned_path = ROOT / "data" / "processed" / "train_cleaned.csv"
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)

    train_cleaned.to_csv(cleaned_path, index=False)

    print("\nCleaned training data saved:", cleaned_path)
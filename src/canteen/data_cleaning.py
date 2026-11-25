import pandas as pd
import yaml
from pathlib import Path

# Load configuration
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

data = params["data"]
settings = params["cleaning"]


def clean_training_data(df):

    df = df.copy()

    # Remove duplicate rows
    if settings["remove_duplicates"]:
        df = df.drop_duplicates().reset_index(drop=True)

    # Correct queue duration
    mask = df["block"] == settings["unit_mixup_group"]

    df.loc[mask, "queue_minutes_yesterday"] = (
        df.loc[mask, "queue_minutes_yesterday"]
        / settings["unit_conversion_factor"]
    )

    # Replace invalid temperature values
    df["temp_c"] = df["temp_c"].replace(
        settings["temperature_sentinel"], float("nan")
    )

    # Correct target typos
    mask = df["plates_sold"] > settings["target_typo_threshold"]

    df.loc[mask, "plates_sold"] = (
        df.loc[mask, "plates_sold"]
        // settings["target_correction_factor"]
    )

    # Standardize category names
    df["day_type"] = df["day_type"].str.strip().str.lower()

    df["day_type"] = df["day_type"].replace(
        settings["day_type_mapping"]
    )

    return df


if __name__ == "__main__":

    train = pd.read_csv(data["train_path"])
    cleaned = clean_training_data(train)

    print("Original shape:", train.shape)
    print("Cleaned shape:", cleaned.shape)
    print("Remaining duplicates:", cleaned.duplicated().sum())

    print("\nQUEUE STATISTICS")
    print(cleaned.groupby("block")["queue_minutes_yesterday"].describe())

    print("\nTEMPERATURE VALIDATION")
    print("Invalid temperatures:",
          (cleaned["temp_c"] == settings["temperature_sentinel"]).sum())
    print("Missing temperatures:", cleaned["temp_c"].isna().sum())

    print("\nTARGET VALIDATION")
    print("Remaining target typos:",
          (cleaned["plates_sold"] > settings["target_typo_threshold"]).sum())
    print("Maximum plates sold:", cleaned["plates_sold"].max())

    print("\nCATEGORY VALIDATION")
    print(cleaned["day_type"].unique())
    print("Total categories:", cleaned["day_type"].nunique())

    # Save the cleaned dataset
       # Create the output folder if it does not exist
    Path(data["cleaned_train_path"]).parent.mkdir(
        parents=True, exist_ok=True
    )

    # Save the cleaned dataset
    cleaned.to_csv(data["cleaned_train_path"], index=False)
    print("\nCleaned dataset saved successfully!")
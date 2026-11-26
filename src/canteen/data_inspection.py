from pathlib import Path

import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

def find_project_root():
    """
    Find the project root by looking for data/raw.
    Works even if this script is executed from another directory.
    """
    current = Path(__file__).resolve().parent

    for path in [current, *current.parents]:
        if (path / "data" / "raw").exists():
            return path

    raise FileNotFoundError(
        "Could not find project root containing data/raw."
    )


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_PATH = DATA_DIR / "train.csv"
LATER_PATH = DATA_DIR / "later.csv"
HOLDOUT_PATH = DATA_DIR / "holdout_features.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_datasets():
    """Load train, later, and holdout datasets."""

    required_files = [TRAIN_PATH, LATER_PATH, HOLDOUT_PATH]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(
                f"Missing file: {file}\n"
                "Check whether the DVC data has been pulled."
            )

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    return train, later, holdout


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

def inspect_dataset(name, df):
    """Print basic information about a dataset."""

    print(f"\n{'=' * 60}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 60}")

    print("\nShape:", df.shape)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())


# ============================================================
# 4. TRAIN DATA STATISTICS
# ============================================================

def inspect_train_statistics(train):
    """Display statistical summaries of the training data."""

    print("\n" + "=" * 60)
    print("TRAIN DATA STATISTICS")
    print("=" * 60)

    print(train.describe().to_string())


# ============================================================
# 5. CATEGORICAL COLUMN CHECKS
# ============================================================

def inspect_categories(train, later):
    """Inspect categorical values in train and later datasets."""

    print("\n" + "=" * 60)
    print("CATEGORICAL VALUES")
    print("=" * 60)

    for column in ["day_type", "block"]:

        if column not in train.columns:
            continue

        print(f"\n{column} - TRAIN:")
        print(train[column].value_counts(dropna=False))

        if column in later.columns:
            print(f"\n{column} - LATER:")
            print(later[column].value_counts(dropna=False))


# ============================================================
# 6. COLUMN DIFFERENCES / POTENTIAL LEAKAGE
# ============================================================

def inspect_column_differences(train, holdout):
    """
    Compare train and holdout columns.

    plates_sold is expected to be the target column,
    so it is excluded from the potential leakage comparison.
    """

    print("\n" + "=" * 60)
    print("COLUMN DIFFERENCES / POTENTIAL LEAKAGE")
    print("=" * 60)

    train_columns = set(train.columns)
    holdout_columns = set(holdout.columns)

    extra_train_columns = (
        train_columns - holdout_columns - {"plates_sold"}
    )

    missing_train_columns = holdout_columns - train_columns

    print("\nExtra training columns (excluding target):")
    print(extra_train_columns)

    print("\nColumns present in holdout but not train:")
    print(missing_train_columns)

    if extra_train_columns:
        print(
            "\nWARNING: Investigate these extra training columns "
            "for potential leakage or schema mismatch."
        )
    else:
        print("\nNo unexpected extra training columns detected.")


# ============================================================
# 7. SENTINEL VALUE CHECK
# ============================================================

def inspect_sentinel_values(train, later):
    """Check for the known temperature sentinel value 999."""

    print("\n" + "=" * 60)
    print("SENTINEL VALUE CHECK")
    print("=" * 60)

    sentinel = 999

    for name, df in [("TRAIN", train), ("LATER", later)]:

        if "temp_c" not in df.columns:
            continue

        count = (df["temp_c"] == sentinel).sum()

        print(f"{name} - temp_c == {sentinel}: {count}")


# ============================================================
# 8. DUPLICATE CHECK
# ============================================================

def inspect_duplicates(train):
    """Inspect duplicate rows."""

    print("\n" + "=" * 60)
    print("DUPLICATE ROW CHECK")
    print("=" * 60)

    duplicate_count = train.duplicated().sum()

    print("Duplicate rows:", duplicate_count)

    # Remove duplicates only for investigation.
    # The original train DataFrame is not modified.
    check = train.drop_duplicates().copy()

    print("Rows before duplicate removal:", len(train))
    print("Rows after duplicate removal:", len(check))

    return check


# ============================================================
# 9. TARGET ANOMALY CHECK
# ============================================================

def inspect_target_anomalies(train, later):
    """
    Investigate unusually large target values.

    The threshold of 1000 is an investigation threshold,
    not an automatic correction rule.
    """

    print("\n" + "=" * 60)
    print("TARGET ANOMALY CHECK")
    print("=" * 60)

    target = "plates_sold"
    threshold = 1000

    if target not in train.columns:
        print(f"{target} not found in training dataset.")
        return

    suspicious = train[train[target] > threshold]

    print(f"\nTrain values above {threshold}:")
    print("Count:", len(suspicious))

    if not suspicious.empty:
        print(
            suspicious.sort_values(
                target, ascending=False
            ).head(20).to_string()
        )

    if target in later.columns:
        print("\nLater target statistics:")
        print(later[target].describe().to_string())


# ============================================================
# 10. B-BLOCK QUEUE UNIT INVESTIGATION
# ============================================================

def inspect_queue_units(train, later):
    """
    Compare queue-time distributions across blocks.

    Tests the proposed B-Block seconds-to-minutes conversion
    without modifying the original datasets.
    """

    print("\n" + "=" * 60)
    print("QUEUE-TIME UNIT INVESTIGATION")
    print("=" * 60)

    column = "queue_minutes_yesterday"

    for name, df in [("TRAIN", train), ("LATER", later)]:

        if column not in df.columns or "block" not in df.columns:
            continue

        print(f"\n{name} - BEFORE correction:")
        print(
            df.groupby("block")[column]
        .describe()
        .round(2)
        .to_string()
        )

        corrected = df.copy()

        corrected.loc[
            corrected["block"] == "B-Block",
            column
        ] /= 60

        print(f"\n{name} - AFTER hypothetical B-Block correction:")
        print(
            corrected.groupby("block")[column]
            .describe()
            .round(2)
            .to_string()
        )


# ============================================================
# 11. STUCK TEMPERATURE SENSOR CHECK
# ============================================================

def inspect_stuck_temperature(check, threshold=4):
    """
    Find consecutive repeated temperature readings.

    Runs are checked separately by block and in date order.
    Only temp_c is treated as a sensor-like feature.
    """

    print("\n" + "=" * 60)
    print("STUCK TEMPERATURE SENSOR CHECK")
    print("=" * 60)

    if "temp_c" not in check.columns:
        print("temp_c column not found.")
        return

    sort_columns = ["date"]

    if "record_id" in check.columns:
        sort_columns.append("record_id")

    for block, group in check.groupby("block"):

        group = group.sort_values(sort_columns).copy()

        # A new run begins whenever temperature changes.
        run_id = group["temp_c"].ne(
            group["temp_c"].shift()
        ).cumsum()

        runs = group.groupby(run_id).agg(
            value=("temp_c", "first"),
            count=("temp_c", "size"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )

        suspicious = runs[runs["count"] >= threshold]

        print(f"\nBlock: {block}")

        if suspicious.empty:
            print(
                f"No temperature runs of "
                f"{threshold} or more consecutive records."
            )
        else:
            print(suspicious.to_string())


# ============================================================
# 12. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("MLOPS DATA INSPECTION")
    print("=" * 60)

    train, later, holdout = load_datasets()

    # Basic inspection
    inspect_dataset("train", train)
    inspect_dataset("later", later)
    inspect_dataset("holdout", holdout)

    # Train statistics
    inspect_train_statistics(train)

    # Categories
    inspect_categories(train, later)

    # Schema / leakage investigation
    inspect_column_differences(train, holdout)

    # Sentinel values
    inspect_sentinel_values(train, later)

    # Duplicate investigation
    check = inspect_duplicates(train)

    # Target anomalies
    inspect_target_anomalies(train, later)

    # Unit investigation
    inspect_queue_units(train, later)

    # Stuck temperature sensor
    inspect_stuck_temperature(check)

    print("\n" + "=" * 60)
    print("DATA INSPECTION COMPLETED")
    print("=" * 60)


# ============================================================
# 13. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
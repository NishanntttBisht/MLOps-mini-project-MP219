import pandas as pd

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

print("\nTRAIN DATASET")
print(train.head())
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())

print("\nLATER DATASET")
print(later.head())
print("Shape:", later.shape)
print("Columns:", later.columns.tolist())

print("\nHOLDOUT DATASET")
print(holdout.head())
print("Shape:", holdout.shape)
print("Columns:", holdout.columns.tolist())

print("\nDATA TYPES")
print(train.dtypes)

print("\nMISSING VALUES")
print(train.isnull().sum())

print("\nDUPLICATE ROWS")
print(train.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(train.describe().to_string())

print("\nDAY TYPE VALUES")
print(train["day_type"].value_counts())

print("\nBLOCK VALUES")
print(train["block"].value_counts())

print("\nEXTRA TRAINING COLUMNS")
print(set(train.columns) - set(holdout.columns) - {"plates_sold"})


# 1. Check for data leakage
print("\n1. DATA LEAKAGE")
print(set(train.columns) - set(holdout.columns) - {"plates_sold"})


# 2. Check for unit mix-up
print("\n2. UNIT MIX-UP")
print(train.groupby("block")["queue_minutes_yesterday"].describe())
print(later.groupby("block")["queue_minutes_yesterday"].describe())


# 3. Check sentinel values
print("\n3. SENTINEL VALUES")
print("Temperature 999 count:", (train["temp_c"] == 999).sum())


# 4. Check duplicate rows
print("\n4. DUPLICATE ROWS")
print("Duplicate count:", train.duplicated().sum())


# 5. Check repeated sensor readings
print("\n5. STUCK SENSOR")

for col in train.select_dtypes(include="number").columns:
    print("\nColumn:", col)
    print(train[col].value_counts().head(5))


# 6. Check target typos
print("\n6. TARGET TYPOS")
print(train["plates_sold"].sort_values(ascending=False).head(40).to_string())

print("\nLater target statistics")
print(later["plates_sold"].describe())


# 7. Check spelling variants
print("\n7. SPELLING VARIANTS")
print(train["day_type"].unique())
print(later["day_type"].unique())


# Remove duplicates temporarily for investigation
check = train.drop_duplicates().sort_values(
    ["date", "record_id"]
)


# Find consecutive repeated readings
print("\nSTUCK SENSOR INVESTIGATION")

for col in check.select_dtypes(include="number").columns:

    groups = check[col].ne(check[col].shift()).cumsum()

    runs = check.groupby(groups).agg(
        value=(col, "first"),
        count=(col, "size"),
        start_date=("date", "min"),
        end_date=("date", "max")
    )

    print("\nColumn:", col)
    print(runs.sort_values("count", ascending=False).head(3))


# Verify queue duration conversion
print("\nCORRECTED B-BLOCK QUEUE")

queue = check.loc[
    check["block"] == "B-Block",
    "queue_minutes_yesterday"
]

print((queue / 60).describe())


# Verify counts after duplicate removal
print("\nFINAL COUNTS")

print("Sentinel values:", (check["temp_c"] == 999).sum())
print("Target typos:", (check["plates_sold"] > 1000).sum())
print("Remaining rows:", len(check))


# Check stuck sensors separately for each block
print("\nFINAL STUCK SENSOR CHECK")

for block, group in check.groupby("block"):

    print("\nBlock:", block)

    for col in check.select_dtypes(include="number").columns:

        runs = group[col].ne(group[col].shift()).cumsum()

        result = group.groupby(runs).agg(
            value=(col, "first"),
            count=(col, "size"),
            start_date=("date", "min"),
            end_date=("date", "max")
        )

        print(col)
        print(result.nlargest(1, "count").to_string())
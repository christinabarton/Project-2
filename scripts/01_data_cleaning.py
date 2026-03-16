# =============================================================================
# Script 01: Data Cleaning and Preprocessing
# =============================================================================
# Project: Predicting Wildfires from Time Series Weather Data
# Authors: Christina Barton, Abby Goss, Rohan Kohli
# Course:  DS 4002, Spring 2026
# Date:    March 2026
#
# Description:
#   This script loads the raw California weather/wildfire dataset, inspects it
#   for missing or faulty values, encodes categorical variables, and saves a
#   clean version of the data to DATA/processed_data.csv for use in downstream
#   scripts.
#
# Input:  DATA/CA_Weather_Fire_Dataset_1984-2025.csv
# Output: DATA/processed_data.csv
# =============================================================================

import pandas as pd
import os

# ---------------------------------------------------------------------------
# 1. Load the raw dataset
# ---------------------------------------------------------------------------
# Build paths relative to this script's location so the project is portable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "DATA")

raw_path = os.path.join(DATA_DIR, "CA_Weather_Fire_Dataset_1984-2025.csv")
df = pd.read_csv(raw_path)

print("=== Raw Data Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n=== Column Names ===")
print(df.columns.tolist())

print("\n=== First 3 Rows ===")
print(df.head(3))

# ---------------------------------------------------------------------------
# 2. Inspect for missing / faulty values
# ---------------------------------------------------------------------------
print("\n=== Missing Values Per Column ===")
print(df.isnull().sum())

# Drop any rows that have at least one null value
before = len(df)
df = df.dropna()
after = len(df)
print(f"\nDropped {before - after} rows with missing values. Remaining: {after}")

# ---------------------------------------------------------------------------
# 3. Parse the DATE column and sort chronologically
# ---------------------------------------------------------------------------
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

print(f"\nDate range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")

# ---------------------------------------------------------------------------
# 4. Encode FIRE_START_DAY (True/False → 1/0)
# ---------------------------------------------------------------------------
# The target variable must be numeric for scikit-learn classifiers
df["FIRE_START_DAY"] = df["FIRE_START_DAY"].map({True: 1, False: 0, "True": 1, "False": 0})
df["FIRE_START_DAY"] = df["FIRE_START_DAY"].astype(int)

print(f"\nFIRE_START_DAY value counts:\n{df['FIRE_START_DAY'].value_counts()}")
print(f"Fire proportion: {df['FIRE_START_DAY'].mean():.3f}")

# ---------------------------------------------------------------------------
# 5. Encode SEASON as a numeric ordinal (used as a feature)
# ---------------------------------------------------------------------------
# Seasons map naturally to a cycle; use simple integer encoding here.
# The model will receive this alongside MONTH, which captures finer
# temporal detail.
season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Fall": 3}
df["SEASON_ENC"] = df["SEASON"].map(season_map)

# ---------------------------------------------------------------------------
# 6. Drop columns that are not needed for modeling
# ---------------------------------------------------------------------------
# DATE is kept for time-series train/test splitting but removed before
# fitting the model. DAY_OF_YEAR is redundant given DATE. The raw SEASON
# string column is replaced by SEASON_ENC.
columns_to_drop = ["DAY_OF_YEAR", "SEASON"]
df = df.drop(columns=columns_to_drop)

print(f"\nFinal columns after cleaning:\n{df.columns.tolist()}")

# ---------------------------------------------------------------------------
# 7. Summary statistics for the cleaned dataset
# ---------------------------------------------------------------------------
print("\n=== Summary Statistics ===")
print(df.describe())

# ---------------------------------------------------------------------------
# 8. Save processed data
# ---------------------------------------------------------------------------
out_path = os.path.join(DATA_DIR, "processed_data.csv")
df.to_csv(out_path, index=False)
print(f"\nCleaned data saved to: {out_path}")

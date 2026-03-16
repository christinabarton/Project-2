# =============================================================================
# Script 04: Wildfire Predictions for 2025
# =============================================================================
# Project: Predicting Wildfires from Time Series Weather Data
# Authors: Christina Barton, Abby Goss, Rohan Kohli
# Course:  DS 4002, Spring 2026
# Date:    March 2026
#
# Description:
#   This script uses the trained Random Forest model (saved in script 03) to
#   generate wildfire-risk predictions for 2025. It:
#     1. Filters the processed dataset to 2025 rows (available observations)
#     2. Applies the saved Random Forest model to produce predictions and
#        fire-probability scores for each day
#     3. Retrains the model on all available labeled data (1985–2024) to
#        maximize predictive power before applying it to 2025
#     4. Saves predictions and a visualization to OUTPUT/
#
# Input:  DATA/processed_data.csv
#         DATA/random_forest_model.pkl
# Output: OUTPUT/predictions_2025.csv
#         OUTPUT/fig_predictions_2025.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, pickle

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "DATA")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load processed data and saved model
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"), parse_dates=["DATE"])
print(f"Loaded {len(df)} rows.")

# Load the model bundle saved by script 03
with open(os.path.join(DATA_DIR, "random_forest_model.pkl"), "rb") as f:
    bundle = pickle.load(f)

rf       = bundle["model"]
features = bundle["features"]

print(f"Model loaded. Features: {features}")

# ---------------------------------------------------------------------------
# 2. Retrain model on ALL labeled data (1985–2024) before predicting 2025
# ---------------------------------------------------------------------------
# Using more training data generally improves prediction quality.
# The 2025 rows may have NaN targets if those dates are unlabeled — we
# exclude them from training regardless.
train_df = df[df["YEAR"] <= 2024].dropna(subset=features + ["FIRE_START_DAY"])
X_all    = train_df[features].values
y_all    = train_df["FIRE_START_DAY"].values

rf.fit(X_all, y_all)
print(f"Model retrained on {len(train_df)} rows (1985–2024).")

# ---------------------------------------------------------------------------
# 3. Apply model to 2025 data
# ---------------------------------------------------------------------------
df_2025 = df[df["YEAR"] == 2025].dropna(subset=features).copy()

if len(df_2025) == 0:
    print("No 2025 data found in processed_data.csv — cannot generate predictions.")
else:
    print(f"Generating predictions for {len(df_2025)} days in 2025.")

    X_2025 = df_2025[features].values

    # Predicted class (0 = no fire, 1 = fire) and probability of fire
    df_2025["PREDICTED_FIRE"] = rf.predict(X_2025)
    df_2025["FIRE_PROBABILITY"] = rf.predict_proba(X_2025)[:, 1]

    # If 2025 has known FIRE_START_DAY labels, compute accuracy
    if "FIRE_START_DAY" in df_2025.columns and df_2025["FIRE_START_DAY"].notna().any():
        from sklearn.metrics import accuracy_score, recall_score, f1_score
        y_true_2025 = df_2025["FIRE_START_DAY"].dropna().astype(int)
        idx_labeled = df_2025["FIRE_START_DAY"].notna()
        y_pred_2025 = df_2025.loc[idx_labeled, "PREDICTED_FIRE"]
        print(f"\n2025 Accuracy: {accuracy_score(y_true_2025, y_pred_2025):.4f}")
        print(f"2025 Recall:   {recall_score(y_true_2025, y_pred_2025, zero_division=0):.4f}")
        print(f"2025 F1:       {f1_score(y_true_2025, y_pred_2025, zero_division=0):.4f}")

    # Summary of predictions
    n_fire_days = df_2025["PREDICTED_FIRE"].sum()
    print(f"\nPredicted fire days in 2025: {n_fire_days} out of {len(df_2025)}")

    # Save predictions CSV
    pred_cols = ["DATE", "PREDICTED_FIRE", "FIRE_PROBABILITY"]
    if "FIRE_START_DAY" in df_2025.columns:
        pred_cols.insert(2, "FIRE_START_DAY")
    out_csv = os.path.join(OUTPUT_DIR, "predictions_2025.csv")
    df_2025[pred_cols].to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}")

    # ---------------------------------------------------------------------------
    # 4. Visualize fire-risk probability over time in 2025
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top panel: fire probability as a continuous risk signal
    axes[0].fill_between(df_2025["DATE"], df_2025["FIRE_PROBABILITY"],
                         alpha=0.6, color="#d62728", label="Fire Probability")
    axes[0].axhline(0.5, linestyle="--", color="black", linewidth=0.8,
                    label="Decision Threshold (0.5)")
    axes[0].set_ylabel("Predicted Fire Probability")
    axes[0].set_title("Predicted Wildfire Risk in California — 2025", fontsize=13)
    axes[0].legend(loc="upper right")
    axes[0].set_ylim(0, 1)

    # Bottom panel: binary fire / no-fire predictions
    fire_days    = df_2025[df_2025["PREDICTED_FIRE"] == 1]
    no_fire_days = df_2025[df_2025["PREDICTED_FIRE"] == 0]
    axes[1].scatter(no_fire_days["DATE"], no_fire_days["PREDICTED_FIRE"],
                    color="#1f77b4", s=4, label="No Fire Predicted (0)", alpha=0.7)
    axes[1].scatter(fire_days["DATE"],    fire_days["PREDICTED_FIRE"],
                    color="#d62728", s=8, marker="^", label="Fire Predicted (1)", zorder=3)
    axes[1].set_ylabel("Predicted Class")
    axes[1].set_xlabel("Date")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["No Fire", "Fire"])
    axes[1].legend(loc="upper right")

    # Format x-axis as months
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_predictions_2025.png"), dpi=150)
    plt.close()
    print("Saved: fig_predictions_2025.png")

print("\nPrediction script complete.")

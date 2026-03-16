# =============================================================================
# Script 02: Exploratory Data Analysis (EDA)
# =============================================================================
# Project: Predicting Wildfires from Time Series Weather Data
# Authors: Christina Barton, Abby Goss, Rohan Kohli
# Course:  DS 4002, Spring 2026
# Date:    March 2026
#
# Description:
#   This script performs exploratory data analysis on the cleaned California
#   weather/wildfire dataset. It reproduces and saves the figures that address
#   the five key questions outlined in MI2:
#     1. How do temperature highs and lows change over time?
#     2. What season has the most wildfires, and does that change over time?
#     3. How has wildfire frequency changed over the dataset period?
#     4. How imbalanced is the FIRE_START_DAY variable?
#     5. Are some predictors highly correlated?
#
# Input:  DATA/processed_data.csv
# Output: OUTPUT/fig_temp_trends.png
#         OUTPUT/fig_fires_by_season.png
#         OUTPUT/fig_fires_by_season_year.png
#         OUTPUT/fig_annual_fire_frequency.png
#         OUTPUT/fig_class_imbalance.png
#         OUTPUT/fig_correlation_heatmap.png
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "DATA")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load processed data
df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"), parse_dates=["DATE"])

# Use a clean, readable style for all figures
sns.set_theme(style="whitegrid", palette="muted")

# ---------------------------------------------------------------------------
# Figure 1: Annual average MAX_TEMP and MIN_TEMP trends over time
# ---------------------------------------------------------------------------
# Compute annual averages to smooth day-to-day noise
annual_temp = df.groupby("YEAR")[["MAX_TEMP", "MIN_TEMP"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(annual_temp["YEAR"], annual_temp["MAX_TEMP"], label="Avg Max Temp", marker="o",
        markersize=3, linewidth=1.5, color="#d62728")
ax.plot(annual_temp["YEAR"], annual_temp["MIN_TEMP"], label="Avg Min Temp", marker="o",
        markersize=3, linewidth=1.5, color="#1f77b4")
ax.set_title("Annual Average Maximum and Minimum Temperatures in California (1984–2025)",
             fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°F)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_temp_trends.png"), dpi=150)
plt.close()
print("Saved: fig_temp_trends.png")

# ---------------------------------------------------------------------------
# Figure 2: Total wildfire counts by season (aggregate over all years)
# ---------------------------------------------------------------------------
# Reverse-map encoded season back to string labels for readability
season_labels = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
df["SEASON_LABEL"] = df["SEASON_ENC"].map(season_labels)

fires_only = df[df["FIRE_START_DAY"] == 1]
season_counts = fires_only["SEASON_LABEL"].value_counts().reindex(
    ["Winter", "Spring", "Summer", "Fall"])

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(season_counts.index, season_counts.values,
              color=["#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5"])
ax.bar_label(bars, padding=3)
ax.set_title("Total Wildfire Starts by Season in California (1984–2025)", fontsize=13)
ax.set_xlabel("Season")
ax.set_ylabel("Number of Wildfire Starts")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_fires_by_season.png"), dpi=150)
plt.close()
print("Saved: fig_fires_by_season.png")

# ---------------------------------------------------------------------------
# Figure 3: Wildfire counts by season per year (stacked bar) to show change
#           over time
# ---------------------------------------------------------------------------
fires_season_year = (fires_only.groupby(["YEAR", "SEASON_LABEL"])
                                .size()
                                .unstack(fill_value=0)
                                .reindex(columns=["Winter", "Spring", "Summer", "Fall"],
                                         fill_value=0))

fig, ax = plt.subplots(figsize=(14, 6))
fires_season_year.plot(kind="bar", stacked=True, ax=ax,
                       color=["#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5"])
ax.set_title("Annual Wildfire Starts by Season in California (1984–2025)", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Wildfire Starts")
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
ax.xaxis.set_tick_params(rotation=45)
ax.legend(title="Season", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_fires_by_season_year.png"), dpi=150)
plt.close()
print("Saved: fig_fires_by_season_year.png")

# ---------------------------------------------------------------------------
# Figure 4: Total annual wildfire frequency over time
# ---------------------------------------------------------------------------
annual_fires = fires_only.groupby("YEAR").size().reset_index(name="fire_count")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(annual_fires["YEAR"], annual_fires["fire_count"], color="#ff7f0e", edgecolor="white")
ax.set_title("Annual Wildfire Frequency in California (1984–2025)", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Days with Wildfire Start")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_annual_fire_frequency.png"), dpi=150)
plt.close()
print("Saved: fig_annual_fire_frequency.png")

# ---------------------------------------------------------------------------
# Figure 5: Class imbalance in FIRE_START_DAY
# ---------------------------------------------------------------------------
class_counts = df["FIRE_START_DAY"].value_counts()
labels = ["No Fire (0)", "Fire (1)"]
values = [class_counts.get(0, 0), class_counts.get(1, 0)]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, color=["#1f77b4", "#d62728"])
ax.bar_label(bars, padding=3, fmt="%d")
ax.set_title("Class Distribution of FIRE_START_DAY", fontsize=13)
ax.set_ylabel("Number of Observations")
total = sum(values)
ax.set_ylim(0, max(values) * 1.12)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.03,
            f"({val/total:.1%})", ha="center", fontsize=10, color="gray")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_class_imbalance.png"), dpi=150)
plt.close()
print("Saved: fig_class_imbalance.png")

# ---------------------------------------------------------------------------
# Figure 6: Correlation heatmap of numeric features
# ---------------------------------------------------------------------------
# Select only numeric, model-relevant columns (exclude date/encoded season)
numeric_cols = ["PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
                "TEMP_RANGE", "WIND_TEMP_RATIO", "LAGGED_PRECIPITATION",
                "LAGGED_AVG_WIND_SPEED", "MONTH", "YEAR", "FIRE_START_DAY"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            linewidths=0.5, ax=ax, mask=mask, vmin=-1, vmax=1)
ax.set_title("Correlation Heatmap of Weather Features and Wildfire Target", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: fig_correlation_heatmap.png")

print("\nEDA complete. All figures saved to OUTPUT/")

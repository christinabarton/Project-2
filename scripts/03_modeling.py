# =============================================================================
# Script 03: Model Building and Evaluation
# =============================================================================
# Project: Predicting Wildfires from Time Series Weather Data
# Authors: Christina Barton, Abby Goss, Rohan Kohli
# Course:  DS 4002, Spring 2026
# Date:    March 2026
#
# Description:
#   This script builds and evaluates two classifiers — Logistic Regression
#   (baseline) and Random Forest — to predict wildfire occurrence
#   (FIRE_START_DAY) from California weather data.
#
#   Workflow:
#     1. Load processed data and define features / target
#     2. Time-series train/test split: train 1985–2022, test 2023–2024
#     3. Scale features for logistic regression
#     4. Fit logistic regression; evaluate on test set
#     5. Fit random forest with 5-fold cross-validation; evaluate on test set
#     6. Plot and save: confusion matrices, ROC curves, feature importance,
#        and a summary metrics table
#
# Input:  DATA/processed_data.csv
# Output: OUTPUT/fig_confusion_matrix_lr.png
#         OUTPUT/fig_confusion_matrix_rf.png
#         OUTPUT/fig_roc_curves.png
#         OUTPUT/fig_feature_importance.png
#         OUTPUT/table_model_metrics.csv
#         DATA/random_forest_model.pkl  (saved model for script 04)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             RocCurveDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "DATA")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# 1. Load processed data
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"), parse_dates=["DATE"])
print(f"Loaded {len(df)} rows from processed_data.csv")

# ---------------------------------------------------------------------------
# 2. Define features and target
# ---------------------------------------------------------------------------
# Features: weather measurements and engineered/lagged variables
# YEAR and MONTH capture temporal trends within the data
FEATURE_COLS = [
    "PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "AVG_WIND_SPEED",
    "TEMP_RANGE", "WIND_TEMP_RATIO",
    "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED",
    "MONTH", "YEAR", "SEASON_ENC"
]
TARGET_COL = "FIRE_START_DAY"

# Drop any remaining rows with NaN in the feature or target columns
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# ---------------------------------------------------------------------------
# 3. Time-series train / test split
# ---------------------------------------------------------------------------
# Using a temporal split preserves the time-series structure of the data and
# prevents leakage of future information into the training set.
# Train: 1985–2022 | Test: 2023–2024
train_mask = df["YEAR"] <= 2022
test_mask  = (df["YEAR"] >= 2023) & (df["YEAR"] <= 2024)

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain size: {len(X_train)}  (years <= 2022)")
print(f"Test size:  {len(X_test)}   (years 2023-2024)")
print(f"Train fire rate: {y_train.mean():.3f}")
print(f"Test  fire rate: {y_test.mean():.3f}")

# ---------------------------------------------------------------------------
# 4. Feature scaling (required for logistic regression)
# ---------------------------------------------------------------------------
# Fit scaler ONLY on training data to avoid data leakage
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 5. Logistic Regression (baseline model)
# ---------------------------------------------------------------------------
print("\n--- Logistic Regression ---")

# class_weight='balanced' adjusts for the class imbalance automatically
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_sc, y_train)

y_pred_lr  = lr.predict(X_test_sc)
y_prob_lr  = lr.predict_proba(X_test_sc)[:, 1]

acc_lr  = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
rec_lr  = recall_score(y_test, y_pred_lr, zero_division=0)
f1_lr   = f1_score(y_test, y_pred_lr, zero_division=0)
auc_lr  = roc_auc_score(y_test, y_prob_lr)

print(f"  Accuracy:  {acc_lr:.4f}")
print(f"  Precision: {prec_lr:.4f}")
print(f"  Recall:    {rec_lr:.4f}")
print(f"  F1:        {f1_lr:.4f}")
print(f"  ROC-AUC:   {auc_lr:.4f}")

# ---------------------------------------------------------------------------
# 6. Random Forest with 5-fold cross-validation
# ---------------------------------------------------------------------------
print("\n--- Random Forest ---")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 5-fold stratified cross-validation on TRAINING data to estimate
# out-of-fold recall and confirm model stability before final test evaluation
cv = StratifiedKFold(n_splits=5, shuffle=False)  # shuffle=False preserves time order

cv_recall = cross_val_score(rf, X_train, y_train, cv=cv,
                            scoring="recall", n_jobs=-1)
cv_f1     = cross_val_score(rf, X_train, y_train, cv=cv,
                            scoring="f1", n_jobs=-1)

print(f"  CV Recall (5-fold): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
print(f"  CV F1     (5-fold): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# Train final model on full training set
rf.fit(X_train, y_train)

y_pred_rf  = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]

acc_rf  = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
rec_rf  = recall_score(y_test, y_pred_rf, zero_division=0)
f1_rf   = f1_score(y_test, y_pred_rf, zero_division=0)
auc_rf  = roc_auc_score(y_test, y_prob_rf)

print(f"\n  Test Accuracy:  {acc_rf:.4f}")
print(f"  Test Precision: {prec_rf:.4f}")
print(f"  Test Recall:    {rec_rf:.4f}")
print(f"  Test F1:        {f1_rf:.4f}")
print(f"  Test ROC-AUC:   {auc_rf:.4f}")

# ---------------------------------------------------------------------------
# 7. Save metrics table
# ---------------------------------------------------------------------------
metrics_df = pd.DataFrame({
    "Model":     ["Logistic Regression", "Random Forest"],
    "Accuracy":  [round(acc_lr,  4), round(acc_rf,  4)],
    "Precision": [round(prec_lr, 4), round(prec_rf, 4)],
    "Recall":    [round(rec_lr,  4), round(rec_rf,  4)],
    "F1":        [round(f1_lr,   4), round(f1_rf,   4)],
    "ROC-AUC":   [round(auc_lr,  4), round(auc_rf,  4)],
})
metrics_path = os.path.join(OUTPUT_DIR, "table_model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics table saved to: {metrics_path}")
print(metrics_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 8. Confusion matrices
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, title, filename):
    """Plot and save a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Fire", "Fire"],
                yticklabels=["No Fire", "Fire"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150)
    plt.close()
    print(f"Saved: {filename}")

plot_confusion(y_test, y_pred_lr,
               "Confusion Matrix — Logistic Regression (Test Set 2023–2024)",
               "fig_confusion_matrix_lr.png")

plot_confusion(y_test, y_pred_rf,
               "Confusion Matrix — Random Forest (Test Set 2023–2024)",
               "fig_confusion_matrix_rf.png")

# ---------------------------------------------------------------------------
# 9. ROC curves (both models on same axes)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, y_prob_lr,
                                 name=f"Logistic Regression (AUC={auc_lr:.3f})",
                                 ax=ax, color="#1f77b4")
RocCurveDisplay.from_predictions(y_test, y_prob_rf,
                                 name=f"Random Forest (AUC={auc_rf:.3f})",
                                 ax=ax, color="#d62728")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
ax.set_title("ROC Curves — Logistic Regression vs. Random Forest (Test Set)", fontsize=12)
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_roc_curves.png"), dpi=150)
plt.close()
print("Saved: fig_roc_curves.png")

# ---------------------------------------------------------------------------
# 10. Random Forest feature importance
# ---------------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", color="#ff7f0e", ax=ax)
ax.set_title("Random Forest Feature Importances", fontsize=12)
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_feature_importance.png"), dpi=150)
plt.close()
print("Saved: fig_feature_importance.png")

# ---------------------------------------------------------------------------
# 11. Save the trained Random Forest model and scaler for use in script 04
# ---------------------------------------------------------------------------
model_bundle = {"model": rf, "scaler": scaler, "features": FEATURE_COLS}
model_path = os.path.join(DATA_DIR, "random_forest_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"Model saved to: {model_path}")

print("\nModeling complete.")

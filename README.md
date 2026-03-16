# Predicting Wildfires from Time Series Weather Data

**DS 4002 — Spring 2026**
**Group: Freenor's Fourth Years** — Christina Barton, Abby Goss, Rohan Kohli
**Group Leader:** Christina Barton

---

## Goal

Build a machine learning model that predicts whether or not a wildfire will occur on a given day in California based on daily weather data (temperature, precipitation, wind speed). The models are trained on historical data from 1985–2022 and evaluated on a held-out test set from 2023–2024.

**Research Question:** Can we find seasonal weather patterns and use them to accurately predict wildfires in subsequent years in California?

---

## Section 1: Software and Platform

### Software
- **Python 3.13** — all analysis, modeling, and figure generation

### Required Packages
Install all dependencies with:
```
pip install pandas numpy scikit-learn matplotlib seaborn reportlab pdfplumber
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic regression, random forest, evaluation metrics, cross-validation |
| `matplotlib` | Plotting figures |
| `seaborn` | Statistical visualization (heatmaps, styled charts) |
| `reportlab` | Generating the Data Appendix PDF |
| `pdfplumber` | PDF text extraction (used during project setup) |

### Platform
Developed and tested on **Windows 11 Pro** (also compatible with macOS and Linux).

---

## Section 2: Map of Documentation

```
Project-2/
│
├── README.md                        ← This file
├── LICENSE.md                       ← MIT License
│
├── DATA/
│   ├── CA_Weather_Fire_Dataset_1984-2025.csv   ← Raw dataset (initial data)
│   ├── processed_data.csv                      ← Cleaned dataset (final analyzed data)
│   ├── random_forest_model.pkl                 ← Saved trained Random Forest model
│   └── data_appendix.pdf                       ← Data Appendix (TIER Protocol 4.0)
│
├── SCRIPTS/
│   ├── 01_data_cleaning.py          ← Step 1: Load raw data, clean, encode, save
│   ├── 02_eda.py                    ← Step 2: Exploratory data analysis + figures
│   ├── 03_modeling.py               ← Step 3: Logistic regression + random forest
│   ├── 04_predict_2025.py           ← Step 4: Generate 2025 wildfire predictions
│   └── build_data_appendix.py       ← Helper: Generates DATA/data_appendix.pdf
│
└── OUTPUT/
    ├── fig_temp_trends.png                ← Annual temperature trends over time
    ├── fig_fires_by_season.png            ← Wildfire counts aggregated by season
    ├── fig_fires_by_season_year.png       ← Seasonal fire breakdown per year
    ├── fig_annual_fire_frequency.png      ← Annual wildfire frequency bar chart
    ├── fig_class_imbalance.png            ← Target variable class distribution
    ├── fig_correlation_heatmap.png        ← Feature correlation heatmap
    ├── fig_confusion_matrix_lr.png        ← Logistic regression confusion matrix
    ├── fig_confusion_matrix_rf.png        ← Random forest confusion matrix
    ├── fig_roc_curves.png                 ← ROC curves for both models
    ├── fig_feature_importance.png         ← Random forest feature importances
    ├── fig_predictions_2025.png           ← 2025 wildfire risk predictions
    ├── table_model_metrics.csv            ← Accuracy/precision/recall/F1/AUC table
    └── predictions_2025.csv              ← Day-by-day 2025 predictions
```

---

## Section 3: Instructions for Reproducing Results

Follow these steps in order to reproduce all results from scratch.

### Prerequisites
1. Install Python 3.10 or later from [python.org](https://www.python.org/).
2. Clone this repository:
   ```
   git clone https://github.com/christinabarton/Project-2.git
   cd Project-2
   ```
3. Install required packages:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn reportlab pdfplumber
   ```

### Step 1 — Data Cleaning (`01_data_cleaning.py`)
```
python SCRIPTS/01_data_cleaning.py
```
Loads `DATA/CA_Weather_Fire_Dataset_1984-2025.csv`, removes 12 rows with missing values, encodes `FIRE_START_DAY` as 0/1, encodes the `SEASON` column as a numeric ordinal, and saves `DATA/processed_data.csv`.

### Step 2 — Exploratory Data Analysis (`02_eda.py`)
```
python SCRIPTS/02_eda.py
```
Reads `DATA/processed_data.csv` and produces six figures saved to `OUTPUT/`:
- Temperature trends over time
- Wildfire counts by season (overall and per year)
- Annual wildfire frequency
- Class imbalance of the target variable
- Feature correlation heatmap

### Step 3 — Model Building and Evaluation (`03_modeling.py`)
```
python SCRIPTS/03_modeling.py
```
Reads `DATA/processed_data.csv`, performs a temporal train/test split (train: 1985–2022, test: 2023–2024), trains both a **Logistic Regression** baseline and a **Random Forest** classifier with 5-fold cross-validation, evaluates both models on the test set, and saves:
- Confusion matrices, ROC curves, and feature importance plot to `OUTPUT/`
- Model metrics table to `OUTPUT/table_model_metrics.csv`
- Trained Random Forest model to `DATA/random_forest_model.pkl`

> Note: This script takes approximately 1–3 minutes to run due to the Random Forest cross-validation step.

### Step 4 — 2025 Predictions (`04_predict_2025.py`)
```
python SCRIPTS/04_predict_2025.py
```
Retrains the Random Forest on all available labeled data (1985–2024) and applies it to the 2025 observations in the dataset, producing:
- `OUTPUT/predictions_2025.csv` — predicted fire class and probability for each 2025 day
- `OUTPUT/fig_predictions_2025.png` — visualization of predicted fire risk

### Optional — Rebuild Data Appendix (`build_data_appendix.py`)
```
python SCRIPTS/build_data_appendix.py
```
Regenerates `DATA/data_appendix.pdf` from the processed data and EDA figures. Run after Steps 1 and 2.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 55.7% | 29.1% | **88.7%** | 43.8% | 72.9% |
| Random Forest | **66.2%** | **34.1%** | 78.7% | **47.5%** | **75.2%** |

Both models surpass the project success criteria (>70% recall, >65% accuracy). The Random Forest achieves better overall performance; the Logistic Regression model maximizes recall, which is particularly valuable for a wildfire-prediction use case where missing a fire is more costly than a false alarm.

---

## References

[1] IPCC, "Climate Change 2021: The Physical Science Basis," IPCC, 2021. Available: https://www.ipcc.ch/report/ar6/wg1/. [Accessed: Feb. 25, 2026]

[2] World Health Organization, "Wildfires," www.who.int, 2025. Available: https://www.who.int/health-topics/wildfires#tab=tab_1. [Accessed: Feb. 25, 2026]

[3] J. McKoy, "Death Count for 2025 LA County Wildfires Likely Higher than Records Show, BU Research Finds," Boston University, Aug. 12, 2025. Available: https://www.bu.edu/articles/2025/death-count-california-wildfires-higher-than-recorded/. [Accessed: Feb. 25, 2026]

[4] M. Kulkarni, "Random Forest Algorithm in Machine Learning Explained," Xoriant, Feb. 24, 2026. Available: https://www.xoriant.com/blog/random-forest-algorithm. [Accessed: Feb. 25, 2026]

[5] "Classification: Accuracy, recall, precision, and related metrics | machine learning | google for developers," Google. Available: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall. [Accessed: Mar. 11, 2026]

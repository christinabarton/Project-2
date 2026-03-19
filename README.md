# Predicting Wildfires from Time Series Weather Data

**DS 4002, Spring 2026**
**Group: Freenor's Fourth Years**: Christina Barton, Abby Goss, Rohan Kohli
**Group Leader:** Christina Barton

---

## Repository Overview

This repository contains all materials for a machine learning project aimed at predicting whether a wildfire will occur on a given day in California based on daily weather data (temperature, precipitation, wind speed). Models are trained on historical data from 1984–2022 and evaluated on a test set from 2023.

**Research Question:** Can we find seasonal weather patterns and use them to accurately predict wildfires in subsequent years in California?

---

## Section 1: Software and Platform

### Software
- **Python 3.13**: all analysis, modeling, and figure generation, run via **Jupyter Notebooks**

### Required Packages
Install all dependencies with:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

| Package | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Logistic regression, random forest, evaluation metrics, cross-validation |
| `matplotlib` | Plotting figures |
| `seaborn` | Statistical visualization (heatmaps, boxplots, bar charts) |

### Platform
Developed and tested on **Windows 11 Pro** (also compatible with macOS and Linux).

---

## Section 2: Map of Documentation

```
Project-2/
│
├── README.md                        ← This file
├── LICENSE (1).txt                  ← MIT License
├── .gitignore
│
├── data/
│   ├── CA_Weather_Fire_Dataset_1984-2025.csv   ← Raw dataset
│   ├── cleaned_df.csv                          ← Cleaned dataset used for analysis
│   └── Data Appendix.pdf                       ← Data Appendix (TIER Protocol 4.0)
│
├── scripts/
│   ├── EDA.ipynb                    ← Exploratory data analysis + all EDA figures
│   └── Model.ipynb                  ← Logistic regression + random forest modeling
│
└── output/
    ├── fig_temp_trends.png                   ← Annual average max/min temperature trends
    ├── fig_fires_by_season.png               ← Wildfire counts by season (aggregate)
    ├── fig_fires_by_season_year.png          ← Seasonal wildfire breakdown per year
    ├── fig_annual_fire_frequency.png         ← Annual wildfire frequency over time
    ├── fig_class_imbalance.png               ← Target variable class distribution
    ├── fig_correlation_heatmap.png           ← Feature correlation heatmap
    ├── quantitative_variable_histograms.png  ← Histograms of quantitative variables
    ├── Logistic Regression Confusion Matrix.png  ← Logistic regression confusion matrix
    ├── Random Forest Feature Importance.png      ← Random forest feature importances
    ├── max_temp_vs_avg_wind_speed.png            ← Max temp vs avg wind speed scatter
    ├── maxtemp_fire_nofire.png                   ← Max temp boxplot: fire vs non-fire
    ├── precipitation_fire_nofire.png             ← Precipitation boxplot: fire vs non-fire
    └── windspeed_fire_nofire.png                 ← Wind speed boxplot: fire vs non-fire
    └── ROC_Curve.png                         ← ROC Curve for Logistic Regression and Random Forest
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
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

### Step 1: Exploratory Data Analysis (`scripts/EDA.ipynb`)

Open `scripts/EDA.ipynb` in Jupyter and run all cells from top to bottom. This notebook:
- Loads `data/CA_Weather_Fire_Dataset_1984-2025.csv`
- Cleans the data (handles missing values, encodes variables) and saves `data/cleaned_df.csv`
- Produces all EDA figures saved to `output/`, including temperature trends, wildfire frequency, seasonal breakdowns, class imbalance, correlation heatmap, variable distributions, and fire vs. non-fire comparisons

### Step 2: Modeling (`scripts/Model.ipynb`)

Open `scripts/Model.ipynb` in Jupyter and run all cells from top to bottom. This notebook:
- Loads `data/CA_Weather_Fire_Dataset_1984-2025.csv` and applies the same cleaning steps
- Defines features: `PRECIPITATION`, `MAX_TEMP`, `MIN_TEMP`, `AVG_WIND_SPEED`, `TEMP_RANGE`, `WIND_TEMP_RATIO`, `LAGGED_PRECIPITATION`, `LAGGED_AVG_WIND_SPEED`
- Performs a temporal train/test split: **train on 1984–2022, test on 2023**
- Trains a **Logistic Regression** model with balanced class weights and 5-fold cross-validation
- Trains a **Random Forest** classifier (200 trees, max depth 10)
- Evaluates both models and saves figures to `output/`

> Note: The Random Forest step may take 1–3 minutes to run.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 75.6% | 67.0% | 74.0% | 70.0% | 82% |
| Random Forest | **76.7%** | **70.0%** | 69.5% | **69.8%** | **83%** |

Both models exceed the project success criteria of >70% recall and >65% accuracy. The Logistic Regression model achieves a higher ROC-AUC (81.5%), making it better at ranking fire risk. However, the Random Forest achieves slightly higher accuracy and precision. For wildfire prediction, where missing a fire is more costly than a false alarm, the Logistic Regression's higher recall may be preferred.

---

## References

[1] IPCC, "Climate Change 2021: The Physical Science Basis," IPCC, 2021. Available: https://www.ipcc.ch/report/ar6/wg1/. [Accessed: Feb. 25, 2026]

[2] World Health Organization, "Wildfires," www.who.int, 2025. Available: https://www.who.int/health-topics/wildfires#tab=tab_1. [Accessed: Feb. 25, 2026]

[3] J. McKoy, "Death Count for 2025 LA County Wildfires Likely Higher than Records Show, BU Research Finds," Boston University, Aug. 12, 2025. Available: https://www.bu.edu/articles/2025/death-count-california-wildfires-higher-than-recorded/. [Accessed: Feb. 25, 2026]

[4] M. Kulkarni, "Random Forest Algorithm in Machine Learning Explained," Xoriant, Feb. 24, 2026. Available: https://www.xoriant.com/blog/random-forest-algorithm. [Accessed: Feb. 25, 2026]

[5] "Classification: Accuracy, recall, precision, and related metrics | machine learning | google for developers," Google. Available: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall. [Accessed: Mar. 11, 2026]

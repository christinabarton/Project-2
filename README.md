**DS 4002, Spring 2026** <br>
**Freenor's Fourth Years**: Christina Barton, Abby Goss, Rohan Kohli <br>
**Group Leader:** Christina Barton

# Project 2: Predicting California Wildfires Using Time Series Weather Data and Machine Learning
## Software and Platform
- Utilized Google Colab's Python notebook to complete the code
- Must install pandas, numpy, matplotlib, seaborn, and scikit-learn packages
- This code runs on either Windows or Mac systems
## Documentation Map
Main Branch Folders
- Data
  - CA_Weather_Fire_Dataset_1984-2025.csv (original dataset)
  - cleaned_df.csv (cleaned dataset)
- Output
- Scripts
  - EDA Cleaned.ipynb (exploratory analysis on cleaned data)
  - EDA P2.ipynb (initial exploratory analysis)
  - Model.ipynb (data cleaning, modeling, and evaluation)
## Analysis Instructions
Cleaning the data
- Convert DATE column to datetime format.
- Remove unnecessary columns: MONTH and DAY_OF_YEAR.
- Convert FIRE_START_DAY to binary (1 = fire, 0 = no fire).
- Remove rows with missing values.
- Save cleaned dataset as cleaned_df.csv.

Modeling Steps
1. Define the model inputs and target:
    - Features (X): PRECIPITATION, MAX_TEMP, MIN_TEMP, AVG_WIND_SPEED, TEMP_RANGE, WIND_TEMP_RATIO, LAGGED_PRECIPITATION, LAGGED_AVG_WIND_SPEED
    - Target (y): FIRE_START_DAY (1 = fire, 0 = no fire)

2. Split the data into training and testing sets using a time-based split.
    - Training set: 1985–2022
    - Testing set: 2023

3. Fit a logistic regression model.
    - Use class_weight="balanced" to account for class imbalance.

4. Generate predictions on the test set using logistic regression.

5. Perform 5-fold cross-validation on the training data.
    - Use recall as the evaluation metric.

6. Fit a random forest model.
    - Use 200 estimators and max depth of 10.
    - Use class_weight="balanced" to account for class imbalance.
  
7. Generate predictions on the test set using the random forest model.
   
8. Use the best performing model to generate predictions for 2024.

   
## Model Evaluation

1. Compute classification metrics (Accuracy, Precision, Recall, F1)
   - Recall is most important due to wildfire prediction risk

2. Generate a Confusion Matrix:
   - Visualize true positives, true negatives, false positives, and false negatives.
   - Assess how well the model identifies fire days.

3. Perform Cross Validation:
   - Evaluate recall stability across 5 folds

4. Create an ROC Curve:
    - Compare logistic regression and random forest performance using AUC

5. Create a Feature Importance analysis:
    - Extract feature importance from the random forest model
    - Identify the most influential predictors
  
6. Compare predicted vs actual fire counts for 2023

## References <br>
[1] IPCC, “Climate Change 2021: The Physical Science Basis,” IPCC, 2021. Available: https://www.ipcc.ch/report/ar6/wg1/. [Accessed: Feb. 25, 2026]
[2] World Health Organization, “Wildfires,” www.who.int, 2025. Available: https://www.who.int/health-topics/wildfires#tab=tab_1. [Accessed: Feb. 25, 2026]
[3] J. McKoy, “Death Count for 2025 LA County Wildfires Likely Higher than Records Show, BU Research Finds,” Boston University, Aug. 12, 2025. Available: https://www.bu.edu/articles/2025/death-count-california-wildfires-higher-than-recorded/. [Accessed: Feb. 25, 2026]
[4] M. Kulkarni, “Random Forest Algorithm in Machine Learning Explained,” Xoriant, Feb. 24, 2026. Available: https://www.xoriant.com/blog/random-forest-algorithm. [Accessed: Feb. 25, 2026]
[5] “Classification: Accuracy, recall, precision, and related metrics  |  machine learning  |  google for developers,” Google, Available: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall [Accessed: Mar. 11, 2026]

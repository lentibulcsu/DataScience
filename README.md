# Software Defect Prediction: Exploratory Data Analysis and Model Training

## 1. Project Overview
This document outlines the Exploratory Data Analysis (EDA) and model training process for a software defect prediction dataset. The goal of the project is to classify software modules as defective or non-defective using machine learning models, specifically Random Forest and K-Nearest Neighbors (KNN).

## 2. Dataset Characteristics and Cleaning
The dataset is loaded and initially inspected to understand its structure and data quality.
* **Dataset Shape:** The raw dataset consists of 60,000 rows and 23 columns.
* **Data Types:** The features are primarily numerical, consisting of 19 integer (`int64`) columns and 4 float (`float64`) columns.
* **Data Quality:** There are absolutely no missing values and zero duplicate rows in the dataset.

## 3. Class Imbalance and Preprocessing
An initial distribution check of the target variable (`defect`) revealed a severe class imbalance.
* **Distribution:** There are 58,223 instances of class `1` (defective) and only 1,777 instances of class `0` (non-defective).
* **Balancing Strategy:** To prevent the model from becoming biased towards the majority class, the dataset was balanced via random downsampling.
* **Balanced Dataset:** The resulting training dataframe contains 3,554 rows (an equal split of 1,777 for each class).
* **Holdout Set:** The remaining 56,446 instances of class `1` were stored in an unused dataframe (`df_unused`) to test the model's performance on unseen majority-class data later.

## 4. Exploratory Data Analysis (EDA)

A correlation matrix was generated to identify which software metrics have the strongest relationship with the target variable.
* **Positive Correlations:** The features most highly correlated with defects are `past_defects` (0.681163), `static_analysis_warnings` (0.565778), and `cyclomatic_complexity` (0.479219).
* **Negative Correlations:** The feature `test_coverage` has a notable negative correlation (-0.342167) with defects, suggesting that higher test coverage is associated with fewer defects.
* **Visualizations:** Scatter plots were utilized to visually confirm the relationships between `test_coverage` vs. `defect` and `past_defects` vs. `defect`.

## 5. Model Training: Random Forest

A Random Forest Classifier was trained as the primary predictive model. The balanced dataset was split into training (2,665 samples) and testing (889 samples) sets using a stratified split.
* **Configuration:** The model was instantiated with 100 estimators (`n_estimators=100`).
* **Test Set Performance:** The model achieved perfect metrics on the test set: 1.0 Accuracy, 1.0 Precision, 1.0 Recall, and 1.0 F1-score.
* **Holdout Set Performance:** When evaluated against the 56,446 unused majority-class samples, the model achieved an accuracy of roughly 0.9998 and a perfect precision of 1.0.

### Feature Importance
Both the model's native feature importance and permutation importance confirmed the findings from the EDA phase.
* The top predictive features were `past_defects`, `static_analysis_warnings`, `cyclomatic_complexity`, and `test_coverage`.
* Permutation importance yielded the exact same top four features in the same order.

## 6. Model Training: K-Nearest Neighbors (KNN)

As a secondary approach, a KNN model was trained and evaluated.
* **Scaling:** Because KNN relies on distance metrics, the features were first standardized using `StandardScaler`.
* **Configuration:** The model was set to use 3 neighbors (`n_neighbors=3`).
* **Test Set Performance:** The KNN model achieved an accuracy of approximately 0.86, a precision of 0.96, a recall of 0.75, and an F1-score of 0.84.
* **Holdout Set Performance:** On the unused data, the KNN model reached an accuracy of roughly 0.978 and an F1-score of 0.989.

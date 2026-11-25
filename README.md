# Customer Churn Prediction in Telecommunications

A research project comparing three machine learning models for predicting customer churn on the IBM Telco Customer Churn dataset. Completed as part of the Postgraduate Diploma in Data Analytics at The Independent Institute of Education (IIE).

## Overview

Telecom companies lose money when customers leave, and keeping a customer is much cheaper than finding a new one. This project builds one consistent, leakage safe pipeline and uses it to compare:

* Logistic Regression
* Random Forest
* LightGBM

Each model was tested with four ways of handling the class imbalance (about 26.5% of customers churned):

* No resampling (baseline)
* Class weighting
* SMOTE
* SMOTENC

## Dataset

* Source: IBM Telco Customer Churn (extended version), included as Telco_customer_churn.xlsx
* 7,043 customers and 33 original columns
* 13 columns were removed before modelling, including IDs, location columns and columns that leak the outcome (for example Churn Label)
* The cleaned file used for modelling is cleaned_churn_data.csv (7,043 rows, 20 columns)

## Project Structure

| File | Purpose |
| --- | --- |
| Telco_customer_churn.xlsx | Raw dataset |
| churn_data.ipynb | Exploratory data analysis and the first preprocessing step (dropping excluded columns) |
| cleaned_churn_data.csv | Cleaned dataset used by the modelling notebook |
| churn_modeling.ipynb | Train/test split, pipelines, tuning, evaluation, SHAP and uncertainty checks |
| cv_summary_corrected.csv | Best cross validation scores for every model and strategy |
| paired_differences_corrected.csv | Fold by fold comparison of each strategy against the baseline |
| selected_models_corrected.csv | Final strategy and hyperparameters chosen for each model |
| test_results_corrected.csv | Final test set results for the three selected models |
| run_details_corrected.json | Parameter grids, CV setup, seed and package versions |
| Diagrams/ | Saved charts from both notebooks |

## Method

1. Stratified 80/20 train/test split (seed 42)
2. Imputation, scaling, encoding, resampling and the model are combined in one pipeline, so every step is fitted only on the training part of each fold
3. Each model and strategy pair is tuned with GridSearchCV using repeated stratified 5 fold cross validation (3 repeats, same folds for every model)
4. The strategy with the highest mean CV F1 score is selected for each model
5. The test set is used once, at a fixed threshold of 0.5
6. SHAP (TreeSHAP) explains the LightGBM model
7. Bootstrap confidence intervals and a subgroup check (gender and senior citizen) are reported

## Results (Test Set)

| Model | Strategy | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | SMOTENC | 0.755 | 0.527 | 0.749 | 0.619 | 0.844 |
| Random Forest | Class weight | 0.772 | 0.554 | 0.727 | 0.629 | 0.850 |
| LightGBM | Class weight | 0.758 | 0.530 | 0.786 | 0.633 | 0.853 |

LightGBM scored highest on F1 and ROC AUC, but all three models were very close and the bootstrap intervals overlap, so no model was clearly better. The top churn drivers from SHAP were contract type, dependents, tenure and fiber optic internet service.

## Tech Stack

* Python 3.11.6
* pandas, NumPy, scikit-learn, imbalanced-learn, LightGBM, SHAP, Matplotlib
* Jupyter Notebook in VS Code

## How to Run

1. Clone the repo
2. Install the packages:

   pip install pandas numpy scikit-learn imbalanced-learn lightgbm shap matplotlib openpyxl

3. Run churn_data.ipynb first, then churn_modeling.ipynb

Note: the tuning step in churn_modeling.ipynb runs twelve grid searches and can take up to around half an hour on a normal laptop.

## Author

Saihil Gurupersad

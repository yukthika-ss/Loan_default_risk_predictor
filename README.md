# Loan_default_risk_predictor
End-to-end project for Loan Default Prediction comparing Logistic Regression, Random Forest, and XGBoost with imbalance handling, threshold tuning, SHAP explainability.

This project focuses on real-world decision metrics like recall and F1-score instead of just accuracy, and demonstrates how model performance changes with dataset size and complexity

Loan Default Prediction System
Project Overview

This project builds a machine learning system to predict whether a loan applicant is likely to default.
The focus is not just on accuracy, but on improving recall and F1-score, which are critical in financial risk prediction.

Problem Statement

Loan default prediction is an imbalanced classification problem where:

Majority class → Non-default (safe loans)
Minority class → Default (risky loans)

A model with high accuracy may still fail if it cannot detect defaulters.

Dataset

Source: Lending Club Dataset

Samples used:
~1000 rows (initial experiment)
~7000 rows (full dataset experiment)

Features include:

Loan amount, interest rate, income
Debt-to-income ratio (DTI)
Credit history
Account details

Workflow

1. Data Preprocessing
   
Handled missing values
Converted categorical variables using one-hot encoding
Feature engineering (credit history length)
Scaling applied for Logistic Regression

2. Handling Class Imbalance

Used class_weight='balanced'
Applied threshold tuning for better recall

3. Models Used
   
🔹 Logistic Regression
Baseline + scaled + class-weighted
Threshold tuning applied
Best performing model

🔹 Random Forest
Hyperparameter tuning (depth, leaf, features)
Threshold tuning applied

🔹 XGBoost
Grid search tuning
Imbalance handling using scale_pos_weight
Threshold tuning applied

6. Model Evaluation

Metrics used:

Accuracy
Precision
Recall (important)
F1-score (primary metric)

Key Results

Model              	  Accuracy	    Recall	    F1-score
Logistic Regression	  ~75–77%	      ~60%	      ~0.44 (Best)
Random Forest	        ~70–75%	      ~65%	      ~0.35
XGBoost	              ~75–80%	      ~50%	      ~0.34

⚠️ Important Insight

Baseline model:
Accuracy ≈ 83%
Recall = 0 ❌

Final model:
Slightly lower accuracy
Much higher recall ✅

This makes it practically useful

Why Logistic Regression Performed Best??

Dataset size is relatively small
Feature relationships are mostly linear
Tree-based models require larger datasets to perform better

Model Explainability (SHAP)

Used SHAP to interpret predictions
Identified key features:
Debt-to-income ratio (DTI)
Interest rate
Income
Credit utilization

Conclusion

Accuracy alone is misleading in imbalanced datasets
Logistic Regression provided the best balance of precision and recall
Threshold tuning significantly improved model performance
Simpler models can outperform complex models depending on data

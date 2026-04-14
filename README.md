🌍 Cross-Border Fraud Detection System (XGBoost + FastAPI)

A production-oriented machine learning system for detecting fraudulent cross-border transactions using advanced feature engineering, hyperparameter optimization, threshold tuning, and real-time inference.

Built with:

⚡ XGBoost for high-performance classification
🔬 Optuna for automated hyperparameter tuning
📊 F1-driven threshold optimization
🚀 FastAPI for live prediction API
Overview

Cross-border fraud is a high-risk problem where transaction behavior often looks suspicious due to:

unusual location patterns
device changes
inconsistent transaction gaps
abnormal payment amounts
prior fraud flags

This project is designed to detect those patterns and classify transactions as fraudulent or legitimate with a focus on recall, precision, and practical deployment.

Key Features
Advanced Feature Engineering

The model uses custom feature emphasis to capture fraud-like behavior:

IsFlaggedBefore is boosted to highlight prior risk
DeviceChange is treated as a stronger anomaly signal
IsAmountUsualForUser helps detect amount irregularities across borders

This makes the model more behavior-aware than a basic classifier.

Hyperparameter Optimization

The model is tuned using Optuna with:

80 optimization trials
Stratified K-Fold cross-validation
F1-focused objective function
Threshold Tuning

Instead of using the default 0.5 cutoff, the model:

computes the precision-recall curve
selects the threshold that maximizes F1 score
improves performance on imbalanced fraud data
Evaluation Metrics

The system reports:

  Confusion Matrix
  Classification Report
  ROC-AUC
  ROC Curve
  Production API

A FastAPI endpoint is included for real-time fraud prediction.

Training Workflow
Load preprocessed feature data
Apply fraud-focused feature weighting
Split data into train, validation, and test sets
Tune XGBoost with Optuna
Train the final model
Find the best decision threshold
Evaluate the model
Save model artifacts for deployment

API Usage:
Start the Server
  uvicorn api:app --reload
Endpoint
  POST /predict

  Sample Request
  {
    "Amount": 5000,
    "SenderLocation": "India",
    "ReceiverLocation": "Singapore",
    "IsFlaggedBefore": false,
    "DeviceType": "Mobile",
    "MerchantCategory": "Electronics",
    "DeviceChange": true,
    "TransactionGap": 120,
    "IsAmountUsualForUser": false
  }
  Sample Response
  {
    "prediction": 1,
    "probability": 0.8732,
    "threshold_used": 0.6421
  }
prediction = 1 → Fraud
prediction = 0 → Legitimate

Why This Project Stands Out

  This is not a basic fraud classifier. It is built around cross-border risk signals, where fraud behavior often emerges from:

  geography mismatch
  device inconsistency
  abnormal transfer timing
  amount anomalies
  prior suspicious activity

It also avoids the usual weak setup of:

  blindly using accuracy
  relying on default thresholds
  ignoring class imbalance
Tech Stack
  Python
  XGBoost
  Optuna
  Scikit-learn
  Pandas
  NumPy
  Matplotlib
  Seaborn
  FastAPI
  Joblib
  
Future Improvements

  Add SHAP-based explainability
  Add deployment with Docker
  Add monitoring for drift and anomaly spikes
  Support streaming transaction feeds
  Add geo-risk scoring and country-risk features



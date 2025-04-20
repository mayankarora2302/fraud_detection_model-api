import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (precision_recall_curve, f1_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import shap

# === Load and modify data === #
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv").values.ravel()

# 🎯 Boost feature preference by scaling
feature_weights = {
    'IsFlaggedBefore': 5.0,
    'DeviceChange': 2.0,
    'IsAmountUsualForUser': 2.5
}

for feature, weight in feature_weights.items():
    if feature in X.columns:
        X[feature] *= weight

# === Split data === #
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === Optuna Objective === #
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'eval_metric': 'logloss',
        'random_state': 42
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val_cv = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_probs = model.predict_proba(X_val_cv)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val_cv, y_probs)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
        best_f1 = np.max(f1s)
        f1_scores.append(best_f1)

    return np.mean(f1_scores)

# === Run Optimization === #
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80, show_progress_bar=True)

print(" Best Trial:", study.best_trial.number)
print(" Best F1 Score:", study.best_value)
print(" Best hyperparameters:", study.best_params)

# === Train Final Model === #
best_params = study.best_params
best_params.update({'eval_metric': 'logloss', 'random_state': 42})

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# === Predict & Tune Threshold === #
y_probs = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1s)]
print(f"🔧 Best Threshold: {best_threshold:.4f}")

# === Evaluation Function === #
def evaluate(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print(" Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\n Classification Report:")
    print(classification_report(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_probs)
    print(f" ROC AUC: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

evaluate(y_test, y_probs, threshold=best_threshold)

# === SHAP for Interpretability === #
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# === Save Everything === #
joblib.dump(model, "xgboost_optuna_model.pkl")
with open("best_threshold.txt", "w") as f:
    f.write(str(best_threshold))
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

print("✅ Model, threshold, and best parameters saved.")

import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# === Load Preprocessed Data === #
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv").values.ravel()

# === Boosted Feature Weights === #
if 'IsFlaggedBefore' in X.columns:
    X['IsFlaggedBefore'] *= 5.0  # Boost when True (1)

if 'DeviceChange' in X.columns:
    X['DeviceChange'] = X['DeviceChange'].apply(lambda x: 2.0 if x == 0 else x)

if 'IsAmountUsualForUser' in X.columns:
    X['IsAmountUsualForUser'] = X['IsAmountUsualForUser'].apply(lambda x: 2.5 if x == 0 else x)

# === Train/Val/Test Split === #
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === Optuna Objective === #
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 400),
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

    f1_scores = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        y_probs = model.predict_proba(X_val_fold)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_val_fold, y_probs)
        f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
        f1_scores.append(np.max(f1s))

    return np.mean(f1_scores)

# === Run Optuna === #
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80, show_progress_bar=True)

print("✅ Best Trial:", study.best_trial.number)
print("✅ Best F1 Score:", study.best_value)
print("✅ Best Params:", study.best_params)

# === Train Final Model === #
best_params = study.best_params
best_params.update({'eval_metric': 'logloss', 'random_state': 42})

model = XGBClassifier(**best_params)
model.fit(X_train, y_train)

# === Predict and Threshold Tuning === #
y_probs = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
f1s = 2 * (precision * recall) / (precision + recall + 1e-6)
best_threshold = thresholds[np.argmax(f1s)]

print(f"🎯 Best Threshold: {best_threshold:.4f}")

# === Evaluation === #
def evaluate(y_true, y_probs, threshold):
    y_pred = (y_probs >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred))

    roc_auc = roc_auc_score(y_true, y_probs)
    print(f"💡 ROC AUC: {roc_auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

evaluate(y_test, y_probs, best_threshold)

# === Save Everything === #
print("💾 Saving model and artifacts...")
joblib.dump(model, "xgb_fraud_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
with open("best_threshold.txt", "w") as f:
    f.write(str(best_threshold))
with open("best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

print("✅ All done. Model saved.")

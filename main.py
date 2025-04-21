from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load model, encoders, threshold, and feature order
model = joblib.load("xgb_fraud_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # exact training column order

with open("best_threshold.txt", "r") as f:
    best_threshold = float(f.read().strip())

# Define input schema
class TransactionInput(BaseModel):
    Amount: float
    SenderLocation: str
    ReceiverLocation: str
    IsFlaggedBefore: bool
    DeviceType: str
    MerchantCategory: str
    DeviceChange: bool
    TransactionGap: int
    IsAmountUsualForUser: bool

@app.post("/predict")
def predict(transaction: TransactionInput):
    input_data = transaction.dict()
    df = pd.DataFrame([input_data])

    # Encode categorical columns with label encoders
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                return {"error": f"Invalid value for '{col}': {df[col].values[0]}"}

    # Ensure all columns match training time
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value
    df = df[feature_columns]  # enforce column order

    # Final numeric safety net
    df = df.apply(pd.to_numeric, errors='coerce')

    # Predict probability
    try:
        prob = model.predict_proba(df)[0][1]  # get probability for fraud class
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

    prediction = int(prob >= best_threshold)

    return {
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "threshold_used": best_threshold
    }

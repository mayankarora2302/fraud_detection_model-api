from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("xgboost_optuna_model.pkl")

label_encoders = joblib.load("label_encoders.pkl")

with open("best_threshold.txt", "r") as f:
    best_threshold = float(f.read().strip())

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

    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except Exception as e:
                return {"error": f"Invalid value for '{col}': {df[col].values[0]}"}

    
    df = df.apply(pd.to_numeric, errors='coerce')

    try:
        prob = model.predict(df)[0] 
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

    prediction = int(prob >= best_threshold)

    return {
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "threshold_used": best_threshold
    }

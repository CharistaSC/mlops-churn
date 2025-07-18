# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np
import uvicorn
import joblib
from preprocessing import load_feature_columns, preprocess_input, determine_group_and_trim
import os

app = FastAPI()

# Load pre-trained Random Forest models
rf_both = joblib.load("app_models/20250702_142942_both_serv_rf_model_v1.pkl")
rf_phone = joblib.load("app_models/20250704_220558_only_ph_rf_model_v2.pkl")
rf_internet = joblib.load("app_models/20250704_194421_only_int_rf_model_v2.pkl")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
feature_columns_path = os.path.join(BASE_DIR, "app_models", "feature_columns.pkl")

# Define expected structure of incoming data
class CustomerData(BaseModel):
    data: Dict[str, Any]

@app.post("/predict")
async def predict(customer: CustomerData):
    try:
        incoming_data = customer.data
        df = pd.DataFrame([incoming_data])

        print("Loading model from:", feature_columns_path)
        feature_columns = joblib.load(feature_columns_path)
        df = preprocess_input(df, feature_columns)

        group, df = determine_group_and_trim(df)

        if group == "both":
            model = rf_both
        elif group == "only_phone":
            model = rf_phone
        elif group == "only_internet":
            model = rf_internet
        else:
            return {"error": "Input does not match any known service group."}

        print(f"Group determind as {group}")
        # Get prediction and probability
        probability = model.predict_proba(df)[0][1]  # probability of churn
        prediction = probability >= 0.5

        result = {
            "churn": bool(prediction),
            "probability": float(round(probability, 4)),
            "group": group
        }
        return {"input": incoming_data, "prediction": result}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
 
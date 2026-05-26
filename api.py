from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/churn_model.pkl")

model_columns = joblib.load("models/model_columns.pkl")

@app.get("/")

def home():
    return {"message": "Customer Churn Prediction API"}

@app.post("/predict")

def predict(data: dict):

    input_df = pd.DataFrame([data])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction)
    }
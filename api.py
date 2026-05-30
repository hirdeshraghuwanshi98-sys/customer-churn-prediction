import logging
import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel, Field

# Setup production log directory
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/production.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Enterprise Churn Core Routing Engine",
    description="Validated production endpoint for enterprise subscriber risk classification.",
    version="2.0.0"
)

try:
    model = joblib.load("models/churn_model.pkl")
    model_columns = joblib.load("models/model_columns.pkl")
    logging.info("Model pipeline artifacts initialized successfully.")
except Exception as e:
    logging.critical(f"Fatal Error during model setup sequence: {str(e)}")
    raise RuntimeError(f"Artifact injection error: {str(e)}")

class InferencePayload(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1)
    tenure: int = Field(..., ge=0, le=72)
    MonthlyCharges: float = Field(..., ge=0.0)
    TotalCharges: float = Field(..., ge=0.0)

@app.get("/health")
def health_check():
    return {"status": "green", "engine_operational": True}

@app.post("/predict")
def score_transaction(payload: InferencePayload):
    try:
        raw_data = payload.model_dump()
        input_df = pd.DataFrame([raw_data])
        
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]
        
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        logging.info(f"Inference successfully executed. Score: {probability}")
        return {
            "vulnerability_detected": bool(prediction),
            "attrition_probability_score": round(probability, 4)
        }
    except Exception as e:
        logging.error(f"Inference computation crash sequence: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal inference calculation breakdown.")
    

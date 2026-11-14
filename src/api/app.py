"""FastAPI microservice for customer churn scoring."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
from src.inference.predict import ChurnPredictor
from src.models.train import train_churn_model

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0.0)
    total_charges: float = Field(..., ge=0.0)
    contract_type: str = Field(..., examples=["Month-to-month", "One year", "Two year"])
    internet_service: str = Field(..., examples=["DSL", "Fiber optic", "No"])
    payment_method: str = Field(..., examples=["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    tech_support: str = Field(..., examples=["Yes", "No"])
    paperless_billing: str = Field(..., examples=["Yes", "No"])

predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    model_path = "models/churn_model.joblib"
    if not os.path.exists(model_path):
        train_churn_model(model_path)
    predictor = ChurnPredictor(model_path)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "churn-scoring-api"}

@app.post("/predict")
def predict_churn(payload: CustomerFeatures):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return predictor.predict_single(payload.model_dump())

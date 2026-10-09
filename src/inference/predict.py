"""Batch and real-time inference engine."""
import joblib
import pandas as pd
from typing import Dict, Any, List

class ChurnPredictor:
    def __init__(self, model_path: str = "models/churn_model.joblib"):
        self.pipeline = joblib.load(model_path)

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([record])
        prob = float(self.pipeline.predict_proba(df)[0, 1])
        prediction = int(prob >= 0.5)
        risk_tier = "High" if prob >= 0.7 else "Medium" if prob >= 0.35 else "Low"
        return {
            "churn_prediction": prediction,
            "churn_probability": round(prob, 4),
            "risk_tier": risk_tier
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.predict_single(r) for r in records]

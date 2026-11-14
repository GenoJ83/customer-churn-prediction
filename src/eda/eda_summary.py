"""Automated EDA statistical profile generator."""
import pandas as pd
import json
from src.data.make_dataset import generate_churn_dataset

def run_eda_profiling():
    df = generate_churn_dataset(3000)
    summary = {
        "dataset_shape": list(df.shape),
        "overall_churn_rate": float(df["churn"].mean()),
        "churn_by_contract": df.groupby("contract_type")["churn"].mean().to_dict(),
        "churn_by_internet": df.groupby("internet_service")["churn"].mean().to_dict(),
        "monthly_charges_quantiles": df["monthly_charges"].quantile([0.25, 0.5, 0.75]).to_dict(),
        "tenure_quantiles": df["tenure_months"].quantile([0.25, 0.5, 0.75]).to_dict()
    }
    return summary

if __name__ == "__main__":
    profile = run_eda_profiling()
    print("EDA Profile Summary:")
    print(json.dumps(profile, indent=2))

"""Synthetic telecom customer churn dataset generator."""
import numpy as np
import pandas as pd

def generate_churn_dataset(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    tenure_months = rng.integers(1, 72, size=n_samples)
    monthly_charges = np.round(rng.uniform(18.0, 118.0, size=n_samples), 2)
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15])
    payment_method = rng.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples)
    tech_support = rng.choice(["Yes", "No"], size=n_samples, p=[0.35, 0.65])
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
    total_charges = np.round(tenure_months * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples), 2)

    # Churn probability based on realistic business risk factors
    log_odds = (
        -1.5 
        - 0.05 * tenure_months 
        + 0.02 * monthly_charges
        + (contract_type == "Month-to-month") * 0.9
        + (internet_service == "Fiber optic") * 0.6
        - (tech_support == "Yes") * 0.5
        + (payment_method == "Electronic check") * 0.4
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, size=n_samples) < prob_churn).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "tech_support": tech_support,
        "paperless_billing": paperless_billing,
        "churn": churn
    })
    return df

if __name__ == "__main__":
    df = generate_churn_dataset(5000)
    print(f"Generated {len(df)} samples with churn rate: {df['churn'].mean():.2%}")

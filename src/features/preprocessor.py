"""Scikit-learn feature preprocessing and column transformation pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "contract_type",
    "internet_service",
    "payment_method",
    "tech_support",
    "paperless_billing"
]

def build_preprocessor() -> ColumnTransformer:
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

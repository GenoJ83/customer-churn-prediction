"""Logistic Regression baseline classifier with cross-validation."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.data.make_dataset import generate_churn_dataset
from src.features.preprocessor import build_preprocessor

def evaluate_baseline():
    df = generate_churn_dataset(4000)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    model_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring="roc_auc")
    print(f"Baseline Logistic Regression Mean ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
    return scores.mean()

if __name__ == "__main__":
    evaluate_baseline()

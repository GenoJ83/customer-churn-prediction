"""Core model training pipeline using Gradient Boosting."""
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.data.make_dataset import generate_churn_dataset
from src.features.preprocessor import build_preprocessor

def train_churn_model(output_path: str = "models/churn_model.joblib"):
    df = generate_churn_dataset(6000)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            random_state=42
        ))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, output_path)
    print(f"Trained model saved to {output_path}")
    return pipeline, X_test, y_test

if __name__ == "__main__":
    train_churn_model()

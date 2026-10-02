"""Hyperparameter optimization and randomized search."""
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from src.data.make_dataset import generate_churn_dataset
from src.features.preprocessor import build_preprocessor

def tune_hyperparameters():
    df = generate_churn_dataset(3000)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]

    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", HistGradientBoostingClassifier(random_state=42))
    ])

    param_distributions = {
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.15],
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__min_samples_leaf": [10, 20, 50],
        "classifier__l2_regularization": [0.0, 0.1, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(pipe, param_distributions, n_iter=8, cv=cv, scoring="roc_auc", random_state=42)
    search.fit(X, y)
    print(f"Best ROC-AUC: {search.best_score_:.4f}")
    print(f"Best Params: {search.best_params_}")
    return search.best_params_

if __name__ == "__main__":
    tune_hyperparameters()

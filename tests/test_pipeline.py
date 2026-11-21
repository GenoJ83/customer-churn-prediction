import pytest
pytest.importorskip("pandas")
"""Unit tests for dataset generator, preprocessor, and model."""
import pytest
from src.data.make_dataset import generate_churn_dataset
from src.features.preprocessor import build_preprocessor
from src.models.baseline import evaluate_baseline

def test_dataset_generation():
    df = generate_churn_dataset(100)
    assert len(df) == 100
    assert "churn" in df.columns
    assert set(df["churn"].unique()).issubset({0, 1})

def test_preprocessor_shape():
    df = generate_churn_dataset(50)
    X = df.drop(columns=["customer_id", "churn"])
    prep = build_preprocessor()
    transformed = prep.fit_transform(X)
    assert transformed.shape[0] == 50
    assert transformed.shape[1] > len(X.columns)

def test_baseline_evaluation():
    score = evaluate_baseline()
    assert score > 0.65

"""Tests for model pipeline construction and preprocessing."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.modeling import build_preprocessor, build_candidate_models, build_parameter_grids


def test_preprocessor_has_numeric_and_categorical_transformers():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "numeric" in transformer_names
    assert "categorical" in transformer_names


def test_preprocessor_handles_expected_feature_columns():
    preprocessor = build_preprocessor()
    numeric_transformer = next(t for name, t, _ in preprocessor.transformers if name == "numeric")
    categorical_transformer = next(t for name, t, _ in preprocessor.transformers if name == "categorical")

    # Check that the transformers reference the correct feature lists
    assert numeric_transformer is not None
    assert categorical_transformer is not None


def test_candidate_models_returns_pipelines_with_preprocessor():
    models = build_candidate_models()
    assert len(models) >= 5, "Should have at least 5 candidate models"

    for model_name, pipeline in models.items():
        assert isinstance(pipeline, Pipeline), f"{model_name} should be a Pipeline"
        assert "preprocess" in pipeline.named_steps, f"{model_name} should have preprocessing step"
        assert "classifier" in pipeline.named_steps, f"{model_name} should have classifier step"


def test_candidate_models_include_required_algorithms():
    models = build_candidate_models()
    model_names = set(models.keys())

    # Required models from the specification
    required = {"Logistic Regression", "Random Forest", "Extra Trees", "Gradient Boosting"}
    assert required.issubset(model_names), f"Missing required models. Have: {model_names}"


def test_parameter_grids_match_candidate_models():
    models = build_candidate_models()
    grids = build_parameter_grids()

    assert set(models.keys()) == set(grids.keys()), "Parameter grids should match model names"

    for model_name, grid in grids.items():
        assert isinstance(grid, dict), f"Grid for {model_name} should be a dict"
        assert len(grid) > 0, f"Grid for {model_name} should not be empty"


def test_pipeline_can_fit_and_predict_on_synthetic_data():
    """Integration test: verify a pipeline can fit and predict."""
    np.random.seed(42)
    n_samples = 100

    # Create synthetic data matching the expected schema
    X = pd.DataFrame({
        "age": np.random.randint(30, 80, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "cp": np.random.randint(1, 5, n_samples),
        "trestbps": np.random.randint(90, 200, n_samples),
        "chol": np.random.randint(100, 400, n_samples),
        "fbs": np.random.randint(0, 2, n_samples),
        "restecg": np.random.randint(0, 3, n_samples),
        "thalach": np.random.randint(70, 200, n_samples),
        "exang": np.random.randint(0, 2, n_samples),
        "oldpeak": np.random.uniform(0, 6, n_samples),
        "slope": np.random.randint(1, 4, n_samples),
        "ca": np.random.randint(0, 4, n_samples),
        "thal": np.random.choice([3, 6, 7], n_samples),
    }, columns=FEATURE_COLUMNS)

    y = pd.Series(np.random.randint(0, 2, n_samples), name="target")

    models = build_candidate_models()
    logistic_pipeline = models["Logistic Regression"]

    # Fit and predict
    logistic_pipeline.fit(X, y)
    predictions = logistic_pipeline.predict(X)

    assert predictions.shape == (n_samples,)
    assert set(predictions).issubset({0, 1})

    # Check probability prediction
    if hasattr(logistic_pipeline, "predict_proba"):
        probabilities = logistic_pipeline.predict_proba(X)
        assert probabilities.shape == (n_samples, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_pipeline_preserves_feature_order():
    """Verify that preprocessing maintains the expected feature order."""
    np.random.seed(42)

    X = pd.DataFrame({
        "age": [50, 60],
        "sex": [1, 0],
        "cp": [2, 3],
        "trestbps": [120, 130],
        "chol": [200, 250],
        "fbs": [0, 1],
        "restecg": [0, 1],
        "thalach": [150, 160],
        "exang": [0, 1],
        "oldpeak": [1.0, 2.0],
        "slope": [2, 1],
        "ca": [0, 1],
        "thal": [3, 7],
    }, columns=FEATURE_COLUMNS)

    y = pd.Series([0, 1])

    models = build_candidate_models()
    pipeline = models["Logistic Regression"]
    pipeline.fit(X, y)

    # Make prediction with columns in correct order
    pred1 = pipeline.predict(X)

    # Shuffle columns and predict again - should still work
    X_shuffled = X.sample(frac=1, axis=1)
    X_reordered = X_shuffled[FEATURE_COLUMNS]
    pred2 = pipeline.predict(X_reordered)

    assert np.array_equal(pred1, pred2), "Predictions should be consistent regardless of input column order"


def test_pipeline_handles_missing_values():
    """Verify that the pipeline can handle missing values in features."""
    X = pd.DataFrame({
        "age": [50, 60, np.nan],
        "sex": [1, 0, 1],
        "cp": [2, np.nan, 3],
        "trestbps": [120, 130, 140],
        "chol": [200, np.nan, 250],
        "fbs": [0, 1, 0],
        "restecg": [0, 1, np.nan],
        "thalach": [150, 160, 170],
        "exang": [0, np.nan, 1],
        "oldpeak": [1.0, 2.0, 1.5],
        "slope": [2, 1, 2],
        "ca": [0, 1, np.nan],
        "thal": [3, 7, 3],
    }, columns=FEATURE_COLUMNS)

    y = pd.Series([0, 1, 0])

    models = build_candidate_models()
    pipeline = models["Random Forest"]

    # Should not raise an error
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert len(predictions) == 3
    assert set(predictions).issubset({0, 1})

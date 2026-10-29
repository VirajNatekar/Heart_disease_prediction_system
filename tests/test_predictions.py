"""End-to-end prediction tests."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app import predict
from src.config import FEATURE_COLUMNS


def test_predict_function_with_valid_input():
    """Test the predict function with a valid input dictionary."""
    valid_input = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 220,
        "fbs": 0,
        "restecg": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.1,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_model.classes_ = [0, 1]

    mock_bundle = {"model": mock_model, "model_name": "Test Classifier"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        result = predict(valid_input)

        assert "positive" in result
        assert result["positive"] is True
        assert "raw_score" in result
        assert result["raw_score"] == pytest.approx(0.7)
        assert result["model_name"] == "Test Classifier"


def test_predict_function_returns_negative_class():
    """Test prediction returning negative class."""
    valid_input = {
        "age": 40,
        "sex": 0,
        "cp": 4,
        "trestbps": 110,
        "chol": 180,
        "fbs": 0,
        "restecg": 0,
        "thalach": 180,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])
    mock_model.classes_ = [0, 1]

    mock_bundle = {"model": mock_model, "model_name": "Test Classifier"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        result = predict(valid_input)

        assert result["positive"] is False
        assert result["raw_score"] == pytest.approx(0.15)


def test_predict_raises_error_when_model_unavailable():
    """Prediction should fail when model is not loaded."""
    valid_input = {key: 50 if key == "age" else 0 for key in FEATURE_COLUMNS}
    valid_input["oldpeak"] = 0.0

    with patch("app.MODEL_BUNDLE", None):
        with pytest.raises(RuntimeError, match="trained model is unavailable"):
            predict(valid_input)


def test_predict_handles_model_without_predict_proba():
    """Some models may not support probability prediction."""
    valid_input = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 220,
        "fbs": 0,
        "restecg": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.1,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    # Mock model without predict_proba
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    del mock_model.predict_proba  # Ensure it doesn't have this method

    mock_bundle = {"model": mock_model, "model_name": "Simple Classifier"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        result = predict(valid_input)

        assert result["positive"] is True
        assert result["raw_score"] is None  # Should be None when probabilities unavailable


def test_predict_respects_feature_order():
    """Prediction should use features in the correct order."""
    valid_input = {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 220,
        "fbs": 0,
        "restecg": 0,
        "thalach": 160,
        "exang": 0,
        "oldpeak": 1.1,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
    mock_model.classes_ = [0, 1]

    mock_bundle = {"model": mock_model, "model_name": "Test Model"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        predict(valid_input)

        # Verify the model was called
        assert mock_model.predict.called

        # Check that the DataFrame was created with the correct column order
        call_args = mock_model.predict.call_args[0]
        df_passed = call_args[0]
        assert isinstance(df_passed, pd.DataFrame)
        assert list(df_passed.columns) == FEATURE_COLUMNS


def test_predict_converts_prediction_to_native_python_types():
    """Result should use native Python types, not numpy types."""
    valid_input = {key: 0 for key in FEATURE_COLUMNS}
    valid_input["age"] = 50
    valid_input["oldpeak"] = 0.0

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1], dtype=np.int64)
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]], dtype=np.float64)
    mock_model.classes_ = [0, 1]

    mock_bundle = {"model": mock_model, "model_name": "Test"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        result = predict(valid_input)

        # Check types are native Python, not numpy
        assert isinstance(result["positive"], bool)
        assert isinstance(result["raw_score"], float)
        assert isinstance(result["model_name"], str)

"""Tests for Flask routes and application behavior."""

import json
import pytest
from unittest.mock import MagicMock, patch

from app import app as flask_app


@pytest.fixture
def app():
    """Configure Flask app for testing."""
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    return flask_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


def test_index_route_returns_200(client):
    """Home page should load successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Heartwise" in response.data
    assert b"heart" in response.data.lower()


def test_health_endpoint_without_model(client):
    """Health endpoint should report model status."""
    with patch("app.MODEL_BUNDLE", None):
        response = client.get("/health")
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        assert "status" in data
        assert "model_ready" in data


def test_health_endpoint_with_model(client):
    """Health endpoint should report ready when model is loaded."""
    mock_bundle = {"model": MagicMock(), "model_name": "Test Model"}
    with patch("app.MODEL_BUNDLE", mock_bundle):
        response = client.get("/health")
        data = json.loads(response.data)
        assert data["model_ready"] is True


def test_predict_route_rejects_missing_fields(client):
    """Form submission with missing fields should be rejected."""
    response = client.post("/predict", data={"age": "50"})
    assert response.status_code == 400
    assert b"required" in response.data.lower()


def test_predict_route_rejects_invalid_numeric_values(client):
    """Out-of-range numeric values should be rejected."""
    data = {
        "age": "7",  # Too young
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        "thal": "3",
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 400
    assert b"Age must be between" in response.data or b"between" in response.data.lower()


def test_predict_route_rejects_invalid_categorical_values(client):
    """Invalid categorical options should be rejected."""
    data = {
        "age": "50",
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        "thal": "99",  # Invalid value
    }
    response = client.post("/predict", data=data)
    assert response.status_code == 400


def test_predict_route_returns_error_when_model_missing(client):
    """Prediction should fail gracefully when model is not trained."""
    valid_data = {
        "age": "52",
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        "thal": "3",
    }

    with patch("app.MODEL_BUNDLE", None):
        response = client.post("/predict", data=valid_data)
        assert response.status_code == 400
        assert b"not trained" in response.data.lower() or b"model" in response.data.lower()


def test_predict_route_with_valid_input_and_mock_model(client):
    """Valid submission should return prediction when model is loaded."""
    valid_data = {
        "age": "52",
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        "thal": "3",
    }

    # Create a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.7, 0.3]]
    mock_model.classes_ = [0, 1]

    mock_bundle = {"model": mock_model, "model_name": "Test Model"}

    with patch("app.MODEL_BUNDLE", mock_bundle):
        response = client.post("/predict", data=valid_data)
        assert response.status_code == 200
        assert b"result" in response.data.lower() or b"prediction" in response.data.lower()


def test_index_displays_form_fields(client):
    """Index page should render all required form fields."""
    response = client.get("/")
    html = response.data.decode("utf-8")

    # Check for key field IDs
    required_fields = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                      "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    for field in required_fields:
        assert f'id="{field}"' in html or f"name=\"{field}\"" in html, f"Field {field} not found in form"


def test_security_headers_are_present(client):
    """Security headers should be set on responses."""
    response = client.get("/")
    headers = response.headers

    assert "X-Content-Type-Options" in headers
    assert "X-Frame-Options" in headers
    assert headers.get("X-Frame-Options") == "DENY"


def test_form_preserves_submitted_values_on_error(client):
    """When validation fails, form should preserve user input."""
    data = {
        "age": "52",
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        # "thal" intentionally missing
    }

    response = client.post("/predict", data=data)
    html = response.data.decode("utf-8")

    # Should preserve the age value
    assert 'value="52"' in html or "52" in html


def test_index_displays_medical_disclaimer(client):
    """Medical disclaimer should be prominently displayed."""
    response = client.get("/")
    html = response.data.decode("utf-8").lower()

    assert "not a diagnosis" in html or "educational" in html
    assert "disclaimer" in html or "important" in html


def test_404_returns_404_status():
    """Nonexistent routes should return 404."""
    client = flask_app.test_client()
    response = client.get("/nonexistent-page-12345")
    assert response.status_code == 404


def test_index_mode_switch_to_brfss(client):
    """Switching mode to brfss should render survey form."""
    response = client.get("/?mode=brfss")
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "Survey-Based Heart Disease Status Model" in html
    assert 'name="GeneralHealth"' in html or 'id="GeneralHealth"' in html
    assert 'name="BMI"' in html or 'id="BMI"' in html


def test_predict_brfss_route_success(client):
    """Valid submission to /predict/brfss should return 200 and prediction card."""
    valid_survey = {
        "Sex": "Female",
        "AgeCategory": "55-59",
        "BMI": "26.4",
        "GeneralHealth": "Good",
        "PhysicalHealthDays": "2",
        "MentalHealthDays": "0",
        "PhysicalActivity": "Yes",
        "SleepHours": "7",
        "Smoking": "No",
        "AlcoholDrinking": "No",
        "Stroke": "No",
        "Diabetes": "No",
        "Asthma": "No",
        "KidneyDisease": "No",
        "DifficultyWalking": "No",
    }
    response = client.post("/predict/brfss", data=valid_survey)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "MODEL A: LARGE-SCALE CDC SURVEY MODEL" in html or "Survey Association" in html or "Pattern Not Flagged" in html or "Flagged" in html


def test_predict_brfss_route_validation_error(client):
    """Invalid submission to /predict/brfss should return 400."""
    response = client.post("/predict/brfss", data={"BMI": "100"})
    assert response.status_code == 400
    html = response.data.decode("utf-8")
    assert "required" in html.lower()


def test_predict_clinical_route_success(client):
    """Valid submission to /predict/clinical should return 200 and result."""
    valid_clinical = {
        "age": "52",
        "sex": "1",
        "cp": "2",
        "trestbps": "130",
        "chol": "220",
        "fbs": "0",
        "restecg": "0",
        "thalach": "160",
        "exang": "0",
        "oldpeak": "1.1",
        "slope": "2",
        "ca": "0",
        "thal": "3",
    }
    response = client.post("/predict/clinical", data=valid_clinical)
    assert response.status_code == 200
    html = response.data.decode("utf-8")
    assert "CLINICAL FEATURE BENCHMARK" in html or "Disease" in html


def test_health_endpoint_reports_both_models(client):
    """Health endpoint reports both CDC BRFSS and UCI models."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "models" in data
    assert "uci_clinical" in data["models"]
    assert "cdc_brfss" in data["models"]

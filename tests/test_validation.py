"""Tests for input validation across both clinical and survey forms."""

from src.config import BRFSS_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.validation import validate_brfss_form, validate_form, validate_uci_form


def valid_uci_form():
    return {
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


def valid_brfss_form():
    return {
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


def test_valid_uci_form_is_parsed_into_expected_schema():
    values, errors = validate_uci_form(valid_uci_form())
    assert errors == []
    assert list(values) == FEATURE_COLUMNS
    assert values["age"] == 52
    assert values["oldpeak"] == 1.1


def test_uci_missing_fields_are_rejected():
    values, errors = validate_uci_form({"age": "50"})
    assert values == {"age": 50}
    assert len(errors) == len(FEATURE_COLUMNS) - 1


def test_uci_out_of_range_and_invalid_category_are_rejected():
    data = valid_uci_form()
    data["age"] = "7"
    data["thal"] = "4"
    values, errors = validate_uci_form(data)
    assert "age" not in values
    assert "thal" not in values
    assert any("Age must be between" in error for error in errors)
    assert any("thal result" in error.lower() for error in errors)


def test_uci_integer_field_rejects_fractional_value():
    data = valid_uci_form()
    data["age"] = "52.5"
    values, errors = validate_uci_form(data)
    assert "age" not in values
    assert any("whole number" in error for error in errors)


def test_valid_brfss_form_is_parsed():
    values, errors = validate_brfss_form(valid_brfss_form())
    assert errors == []
    assert list(values) == BRFSS_FEATURE_COLUMNS
    assert values["BMI"] == 26.4
    assert values["GeneralHealth"] == "Good"
    assert values["PhysicalHealthDays"] == 2


def test_brfss_missing_fields_are_rejected():
    values, errors = validate_brfss_form({"Sex": "Female"})
    assert values == {"Sex": "Female"}
    assert len(errors) == len(BRFSS_FEATURE_COLUMNS) - 1


def test_brfss_out_of_range_numeric_rejected():
    data = valid_brfss_form()
    data["BMI"] = "5.0"  # below min 12.0
    data["PhysicalHealthDays"] = "45"  # above max 30
    values, errors = validate_brfss_form(data)
    assert "BMI" not in values
    assert "PhysicalHealthDays" not in values
    assert any("BMI must be between" in e for e in errors)
    assert any("Physical health not good" in e for e in errors)


def test_brfss_invalid_categorical_rejected():
    data = valid_brfss_form()
    data["AgeCategory"] = "Unknown"
    data["GeneralHealth"] = "Super"
    values, errors = validate_brfss_form(data)
    assert "AgeCategory" not in values
    assert "GeneralHealth" not in values
    assert len(errors) >= 2

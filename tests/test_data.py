"""Tests for dataset loading, schema verification, and target leakage prevention."""

import pandas as pd
import pytest

from src.config import BRFSS_FEATURE_COLUMNS, FEATURE_COLUMNS
from src.data import (
    check_target_leakage,
    harmonize_brfss_2020,
    harmonize_brfss_2022,
    load_brfss_dataset,
    load_dataset,
    load_uci_dataset,
)


def test_uci_loader_maps_original_outcome_to_binary_and_deduplicates(tmp_path):
    base = {
        "age": 52, "sex": 1, "cp": 2, "trestbps": 130, "chol": 220,
        "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0,
        "oldpeak": 1.1, "slope": 2, "ca": 0, "thal": 3,
    }
    rows = []
    for index in range(60):
        row = dict(base)
        row["age"] = 20 + index
        row["trestbps"] = 100 + (index % 45)
        row["chol"] = 160 + (index * 3)
        row["thalach"] = 100 + (index % 120)
        row["num"] = 0 if index < 30 else (1 if index % 2 else 2)
        rows.append(row)
    rows.append(dict(rows[0]))
    path = tmp_path / "toy_uci.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    X, y, notes = load_uci_dataset(path)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()) == {0, 1}
    assert notes["duplicate_rows_removed"] == 1
    assert len(X) == 60


def test_uci_loader_requires_expected_schema(tmp_path):
    path = tmp_path / "wrong_uci.csv"
    pd.DataFrame({"age": [40], "num": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing columns"):
        load_uci_dataset(path)


def test_brfss_2020_harmonization():
    raw_data = {
        "HeartDisease": ["No", "Yes"],
        "BMI": [24.5, 31.2],
        "Smoking": ["No", "Yes"],
        "AlcoholDrinking": ["No", "No"],
        "Stroke": ["No", "Yes"],
        "PhysicalHealth": [0.0, 10.0],
        "MentalHealth": [5.0, 0.0],
        "DiffWalking": ["No", "Yes"],
        "Sex": ["Female", "Male"],
        "AgeCategory": ["30-34", "80 or older"],
        "Diabetic": ["No", "Yes"],
        "PhysicalActivity": ["Yes", "No"],
        "GenHealth": ["Very good", "Fair"],
        "SleepTime": [7.0, 6.0],
        "Asthma": ["No", "Yes"],
        "KidneyDisease": ["No", "No"],
    }
    raw_df = pd.DataFrame(raw_data)
    X, y = harmonize_brfss_2020(raw_df)

    assert list(X.columns) == BRFSS_FEATURE_COLUMNS
    assert list(y.values) == [0, 1]
    assert X["AgeCategory"].iloc[1] == "80+"
    assert X["Smoking"].iloc[1] == "Yes"
    assert X["GeneralHealth"].iloc[0] == "Very good"


def test_brfss_2022_harmonization_and_target_derivation():
    raw_data = {
        "HadHeartAttack": ["No", "Yes", "No"],
        "HadAngina": ["No", "No", "Yes"],
        "Sex": ["Female", "Male", "Female"],
        "AgeCategory": ["Age 25 to 29", "Age 65 to 69", "Age 80 or older"],
        "BMI": [22.0, 28.5, 33.1],
        "GeneralHealth": ["Excellent", "Good", "Poor"],
        "PhysicalHealthDays": [0.0, 5.0, 20.0],
        "MentalHealthDays": [0.0, 0.0, 15.0],
        "PhysicalActivities": ["Yes", "No", "No"],
        "SleepHours": [8.0, 7.0, 5.0],
        "SmokerStatus": ["Never smoked", "Former smoker", "Current smoker - now smokes every day"],
        "AlcoholDrinkers": ["No", "Yes", "No"],
        "HadStroke": ["No", "No", "Yes"],
        "HadDiabetes": ["No", "Yes", "No, pre-diabetes or borderline diabetes"],
        "HadAsthma": ["No", "No", "Yes"],
        "HadKidneyDisease": ["No", "No", "Yes"],
        "DifficultyWalking": ["No", "No", "Yes"],
    }
    raw_df = pd.DataFrame(raw_data)
    X, y = harmonize_brfss_2022(raw_df)

    assert list(X.columns) == BRFSS_FEATURE_COLUMNS
    assert list(y.values) == [0, 1, 1]  # Either attack or angina -> 1
    assert X["AgeCategory"].tolist() == ["25-29", "65-69", "80+"]
    assert X["Smoking"].tolist() == ["No", "Yes", "Yes"]
    assert X["Diabetes"].tolist() == ["No", "Yes", "Borderline"]


def test_target_leakage_detector_raises_error_on_leakage():
    # If any outcome component is present, check_target_leakage must raise ValueError
    leaked_features = pd.DataFrame({
        "Sex": ["Male"],
        "BMI": [25.0],
        "HadHeartAttack": ["Yes"],  # LEAKAGE!
    })
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        check_target_leakage(leaked_features)

    sample_weight_leakage = pd.DataFrame({
        "Sex": ["Female"],
        "BMI": [24.0],
        "_LLCPWT": [102.5],  # LEAKAGE!
    })
    with pytest.raises(ValueError, match="CRITICAL TARGET LEAKAGE DETECTED"):
        check_target_leakage(sample_weight_leakage)

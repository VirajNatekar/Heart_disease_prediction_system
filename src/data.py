"""Dataset loading, validation, leakage checks, and multi-year harmonization.

Provides loaders for:
- Model A: CDC BRFSS Large-Scale Survey Model (2020 training pool, 2022 temporal holdout)
- Model B: UCI Cleveland Clinical Benchmark (303 records)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    BRFSS_FEATURE_COLUMNS,
    BRFSS_NUMERIC_FEATURES,
    BRFSS_CATEGORICAL_FEATURES,
    BRFSS_TARGET_COLUMN,
    UCI_FEATURE_COLUMNS,
    UCI_TARGET_COLUMN,
)

# Canonical age order
AGE_CATEGORIES = [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"
]

# Leakage variables that MUST NEVER be included in BRFSS predictor matrices
LEAKAGE_TARGET_COMPONENTS = {
    "HeartDisease",
    "HadHeartAttack",
    "HadAngina",
    "CVDINFR4",
    "CVDCRHD4",
    "_MICHD",
    "_LLCPWT",
    "target",
}


def check_target_leakage(features: pd.DataFrame, target_name: str = "HeartDisease") -> None:
    """Ensure no outcome components, direct proxies, or sample weights exist in predictors."""
    feature_cols = set(features.columns)
    leaked = feature_cols.intersection(LEAKAGE_TARGET_COMPONENTS)
    if leaked:
        raise ValueError(
            f"CRITICAL TARGET LEAKAGE DETECTED: The feature set contains derived outcome "
            f"components or sample weights: {leaked}. These must never be used as predictors."
        )


def harmonize_brfss_2020(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Harmonize the CDC BRFSS 2020 dataset into the canonical feature schema."""
    df = raw_df.copy()

    # Target: HeartDisease ('Yes' -> 1, 'No' -> 0)
    if "HeartDisease" not in df.columns:
        raise ValueError("BRFSS 2020 dataset missing 'HeartDisease' target column.")
    y = (df["HeartDisease"].astype(str).str.strip().str.lower() == "yes").astype(int).rename(BRFSS_TARGET_COLUMN)

    # Harmonize features
    features = pd.DataFrame(index=df.index)
    features["Sex"] = df["Sex"].astype(str).str.strip()

    # Age mapping
    features["AgeCategory"] = df["AgeCategory"].astype(str).str.strip().replace({"80 or older": "80+"})

    # Numeric features
    features["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    features["GeneralHealth"] = df["GenHealth"].astype(str).str.strip()
    features["PhysicalHealthDays"] = pd.to_numeric(df["PhysicalHealth"], errors="coerce")
    features["MentalHealthDays"] = pd.to_numeric(df["MentalHealth"], errors="coerce")
    features["PhysicalActivity"] = df["PhysicalActivity"].astype(str).str.strip()
    features["SleepHours"] = pd.to_numeric(df["SleepTime"], errors="coerce")
    features["Smoking"] = df["Smoking"].astype(str).str.strip()
    features["AlcoholDrinking"] = df["AlcoholDrinking"].astype(str).str.strip()
    features["Stroke"] = df["Stroke"].astype(str).str.strip()

    # Diabetes mapping
    diab_map = {
        "No": "No",
        "Yes": "Yes",
        "No, borderline diabetes": "Borderline",
        "Yes (during pregnancy)": "Pregnancy-only",
    }
    features["Diabetes"] = df["Diabetic"].map(diab_map).fillna("No")

    features["Asthma"] = df["Asthma"].astype(str).str.strip()
    features["KidneyDisease"] = df["KidneyDisease"].astype(str).str.strip()
    features["DifficultyWalking"] = df["DiffWalking"].astype(str).str.strip()

    check_target_leakage(features, BRFSS_TARGET_COLUMN)
    return features[BRFSS_FEATURE_COLUMNS], y


def harmonize_brfss_2022(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Harmonize the CDC BRFSS 2022 dataset into the canonical feature schema."""
    df = raw_df.copy()

    # Outcome derivation: HadHeartAttack == 'Yes' OR HadAngina == 'Yes' (matching _MICHD)
    if "HadHeartAttack" not in df.columns or "HadAngina" not in df.columns:
        raise ValueError("BRFSS 2022 dataset missing HadHeartAttack or HadAngina columns.")
    
    had_attack = df["HadHeartAttack"].astype(str).str.strip().str.lower() == "yes"
    had_angina = df["HadAngina"].astype(str).str.strip().str.lower() == "yes"
    y = (had_attack | had_angina).astype(int).rename(BRFSS_TARGET_COLUMN)

    # Harmonize features
    features = pd.DataFrame(index=df.index)
    features["Sex"] = df["Sex"].astype(str).str.strip()

    # 2022 Age category mapping: 'Age 18 to 24' -> '18-24', etc.
    age_map_22 = {
        "Age 18 to 24": "18-24", "Age 25 to 29": "25-29", "Age 30 to 34": "30-34",
        "Age 35 to 39": "35-39", "Age 40 to 44": "40-44", "Age 45 to 49": "45-49",
        "Age 50 to 54": "50-54", "Age 55 to 59": "55-59", "Age 60 to 64": "60-64",
        "Age 65 to 69": "65-69", "Age 70 to 74": "70-74", "Age 75 to 79": "75-79",
        "Age 80 or older": "80+",
    }
    features["AgeCategory"] = df["AgeCategory"].map(age_map_22).fillna("18-24")

    # Numeric features
    features["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    features["GeneralHealth"] = df["GeneralHealth"].astype(str).str.strip()
    features["PhysicalHealthDays"] = pd.to_numeric(df["PhysicalHealthDays"], errors="coerce")
    features["MentalHealthDays"] = pd.to_numeric(df["MentalHealthDays"], errors="coerce")
    features["PhysicalActivity"] = df["PhysicalActivities"].astype(str).str.strip()
    features["SleepHours"] = pd.to_numeric(df["SleepHours"], errors="coerce")

    # SmokerStatus to binary Smoking (ever smoked 100 cigarettes)
    smoke_map = {
        "Never smoked": "No",
        "Former smoker": "Yes",
        "Current smoker - now smokes every day": "Yes",
        "Current smoker - now smokes some days": "Yes",
    }
    features["Smoking"] = df["SmokerStatus"].map(smoke_map).fillna("No")

    features["AlcoholDrinking"] = df["AlcoholDrinkers"].astype(str).str.strip()
    features["Stroke"] = df["HadStroke"].astype(str).str.strip()

    # Diabetes mapping
    diab_map_22 = {
        "No": "No",
        "Yes": "Yes",
        "No, pre-diabetes or borderline diabetes": "Borderline",
        "Yes, but only during pregnancy (female)": "Pregnancy-only",
    }
    features["Diabetes"] = df["HadDiabetes"].map(diab_map_22).fillna("No")

    features["Asthma"] = df["HadAsthma"].astype(str).str.strip()
    features["KidneyDisease"] = df["HadKidneyDisease"].astype(str).str.strip()
    features["DifficultyWalking"] = df["DifficultyWalking"].astype(str).str.strip()

    check_target_leakage(features, BRFSS_TARGET_COLUMN)
    return features[BRFSS_FEATURE_COLUMNS], y


def load_brfss_dataset(
    csv_path: str | Path,
    year: int | None = None,
    sample_size: int | None = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Load, harmonize, and validate CDC BRFSS dataset."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"BRFSS dataset not found at {path}")

    # Infer year if not supplied
    if year is None:
        if "2020" in path.as_posix():
            year = 2020
        elif "2022" in path.as_posix():
            year = 2022
        else:
            year = 2020

    raw_df = pd.read_csv(path, low_memory=False)
    raw_count = len(raw_df)

    if year == 2020:
        X, y = harmonize_brfss_2020(raw_df)
    elif year == 2022:
        X, y = harmonize_brfss_2022(raw_df)
    else:
        raise ValueError(f"Unsupported BRFSS survey year: {year}")

    # Remove records with missing target
    valid_mask = ~y.isna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    # Sample if requested (e.g. for rapid development / unit tests)
    if sample_size and len(X) > sample_size:
        sample_indices = y.groupby(y).sample(
            n=int(sample_size / 2),
            random_state=random_state,
            replace=False,
        ).index
        # If stratified equal sample exceeds positive class, take random stratified sample
        if len(sample_indices) < sample_size:
            X = X.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            y = y.loc[X.index].reset_index(drop=True)
        else:
            X = X.loc[sample_indices].reset_index(drop=True)
            y = y.loc[sample_indices].reset_index(drop=True)

    notes = {
        "dataset_year": year,
        "raw_record_count": raw_count,
        "usable_record_count": len(X),
        "positive_class_count": int((y == 1).sum()),
        "negative_class_count": int((y == 0).sum()),
        "positive_rate": float(y.mean()),
        "missing_features_count": int(X.isna().sum().sum()),
    }
    return X, y, notes


def load_uci_dataset(csv_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """Load a named-column Cleveland CSV and return X, binary y, and data notes."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_dataset.py` first."
        )

    frame = pd.read_csv(path, na_values=["?", "", "NA", "NaN"])
    missing = [column for column in [*UCI_FEATURE_COLUMNS, UCI_TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(
            "Dataset schema is not the expected processed Cleveland format. "
            f"Missing columns: {', '.join(missing)}"
        )

    frame = frame[[*UCI_FEATURE_COLUMNS, UCI_TARGET_COLUMN]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    before = len(frame)
    frame = frame.dropna(subset=[UCI_TARGET_COLUMN])
    frame = frame[frame[UCI_TARGET_COLUMN].between(0, 4)]
    invalid_target_rows_removed = before - len(frame)

    before_dedupe = len(frame)
    frame = frame.drop_duplicates(subset=[*UCI_FEATURE_COLUMNS, UCI_TARGET_COLUMN]).reset_index(drop=True)
    duplicate_rows_removed = before_dedupe - len(frame)

    X = frame[UCI_FEATURE_COLUMNS].copy()
    y = (frame[UCI_TARGET_COLUMN] > 0).astype(int).rename("target")

    if len(frame) < 50 or y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError(
            "The dataset is too small or does not contain both outcome classes."
        )

    notes = {
        "rows_loaded_after_target_validation": int(len(frame)),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "invalid_target_rows_removed": int(invalid_target_rows_removed),
        "missing_feature_values": int(X.isna().sum().sum()),
        "positive_records": int(y.sum()),
        "negative_records": int((y == 0).sum()),
    }
    return X, y, notes


# Backward compatibility alias
load_dataset = load_uci_dataset

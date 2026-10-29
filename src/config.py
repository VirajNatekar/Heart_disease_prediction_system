"""Configuration, schemas, and form metadata for both HeartWise prediction models.

Model A: CDC BRFSS Large-Scale Survey Model
Model B: UCI Cleveland Clinical Benchmark
"""

from __future__ import annotations
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

# ==============================================================================
# MODEL B: UCI CLEVELAND CLINICAL BENCHMARK CONFIGURATION
# ==============================================================================
UCI_CSV_PATH = DATA_DIR / "heart_disease.csv"
UCI_MODEL_PATH = ARTIFACTS_DIR / "uci_clinical_model.joblib"
UCI_METADATA_PATH = ARTIFACTS_DIR / "uci_clinical_metadata.json"

UCI_FEATURE_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

UCI_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
UCI_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
UCI_TARGET_COLUMN = "num"
UCI_BINARY_TARGET = "target"

# Backwards compatibility aliases
FEATURE_COLUMNS = UCI_FEATURE_COLUMNS
NUMERIC_FEATURES = UCI_NUMERIC_FEATURES
CATEGORICAL_FEATURES = UCI_CATEGORICAL_FEATURES
TARGET_COLUMN = UCI_TARGET_COLUMN
BINARY_TARGET_COLUMN = UCI_BINARY_TARGET

UCI_FORM_SECTIONS = [
    {
        "title": "Patient profile",
        "description": "Basic demographic information from clinical records.",
        "fields": ["age", "sex"],
    },
    {
        "title": "Measurements",
        "description": "Enter measured physiological values in the units shown.",
        "fields": ["trestbps", "chol", "fbs", "thalach"],
    },
    {
        "title": "Clinical indicators",
        "description": "Coded clinical findings from stress ECG and imaging tests.",
        "fields": ["cp", "restecg", "exang", "oldpeak", "slope", "ca", "thal"],
    },
]

UCI_FIELD_METADATA = {
    "age": {
        "label": "Age",
        "type": "number",
        "unit": "years",
        "min": 18,
        "max": 100,
        "step": 1,
        "help": "Age in completed years.",
    },
    "sex": {
        "label": "Sex (source coding)",
        "type": "select",
        "help": "Historical dataset binary biological sex recording.",
        "options": {"0": "Female (0)", "1": "Male (1)"},
    },
    "cp": {
        "label": "Chest pain type",
        "type": "select",
        "help": "Clinical category documented during initial evaluation.",
        "options": {
            "1": "Typical angina (1)",
            "2": "Atypical angina (2)",
            "3": "Non-anginal pain (3)",
            "4": "Asymptomatic (4)",
        },
    },
    "trestbps": {
        "label": "Resting blood pressure",
        "type": "number",
        "unit": "mm Hg",
        "min": 70,
        "max": 250,
        "step": 1,
        "help": "Resting blood pressure in mm Hg upon hospital admission.",
    },
    "chol": {
        "label": "Serum cholesterol",
        "type": "number",
        "unit": "mg/dL",
        "min": 80,
        "max": 700,
        "step": 1,
        "help": "Total serum cholesterol in mg/dL.",
    },
    "fbs": {
        "label": "Fasting blood sugar > 120 mg/dL",
        "type": "select",
        "help": "Fasting blood sugar greater than 120 mg/dL.",
        "options": {"0": "No (0)", "1": "Yes (1)"},
    },
    "restecg": {
        "label": "Resting ECG result",
        "type": "select",
        "help": "Resting electrocardiographic results.",
        "options": {
            "0": "Normal (0)",
            "1": "ST-T wave abnormality (1)",
            "2": "Possible/definite LV hypertrophy (2)",
        },
    },
    "thalach": {
        "label": "Maximum heart rate achieved",
        "type": "number",
        "unit": "bpm",
        "min": 50,
        "max": 250,
        "step": 1,
        "help": "Maximum heart rate achieved during exercise stress testing.",
    },
    "exang": {
        "label": "Exercise-induced angina",
        "type": "select",
        "help": "Presence of angina pectoris provoked by exercise test.",
        "options": {"0": "No (0)", "1": "Yes (1)"},
    },
    "oldpeak": {
        "label": "ST depression (oldpeak)",
        "type": "number",
        "unit": "relative units",
        "min": 0.0,
        "max": 10.0,
        "step": 0.1,
        "help": "ST depression induced by exercise relative to rest.",
    },
    "slope": {
        "label": "Peak exercise ST slope",
        "type": "select",
        "help": "Slope of the peak exercise ST segment.",
        "options": {
            "1": "Upsloping (1)",
            "2": "Flat (2)",
            "3": "Downsloping (3)",
        },
    },
    "ca": {
        "label": "Major vessels visualized (ca)",
        "type": "select",
        "help": "Number of major vessels (0-3) colored by fluoroscopy.",
        "options": {"0": "0 vessels", "1": "1 vessel", "2": "2 vessels", "3": "3 vessels"},
    },
    "thal": {
        "label": "Thal result (source code)",
        "type": "select",
        "help": "Historical UCI coding; not a universal clinical standard.",
        "options": {"3": "Normal (3)", "6": "Fixed defect (6)", "7": "Reversible defect (7)"},
    },
}

# Backwards compatibility
FIELD_METADATA = UCI_FIELD_METADATA
FORM_SECTIONS = UCI_FORM_SECTIONS


# ==============================================================================
# MODEL A: CDC BRFSS LARGE-SCALE SURVEY MODEL CONFIGURATION
# ==============================================================================
BRFSS_MODEL_PATH = ARTIFACTS_DIR / "brfss_model.joblib"
BRFSS_METADATA_PATH = ARTIFACTS_DIR / "brfss_model_metadata.json"

BRFSS_PROCESSED_TRAIN_PATH = DATA_DIR / "processed" / "brfss_model" / "brfss_train_2020.csv"
BRFSS_PROCESSED_TEST_PATH = DATA_DIR / "processed" / "brfss_model" / "brfss_test_2022.csv"

BRFSS_FEATURE_COLUMNS = [
    "Sex",
    "AgeCategory",
    "BMI",
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "PhysicalActivity",
    "SleepHours",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "Diabetes",
    "Asthma",
    "KidneyDisease",
    "DifficultyWalking",
]

BRFSS_NUMERIC_FEATURES = ["BMI", "PhysicalHealthDays", "MentalHealthDays", "SleepHours"]
BRFSS_CATEGORICAL_FEATURES = [
    "Sex",
    "AgeCategory",
    "GeneralHealth",
    "PhysicalActivity",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "Diabetes",
    "Asthma",
    "KidneyDisease",
    "DifficultyWalking",
]
BRFSS_TARGET_COLUMN = "HeartDisease"

BRFSS_FORM_SECTIONS = [
    {
        "title": "Demographics & Body Metrics",
        "description": "Baseline demographic and physical characteristics.",
        "fields": ["Sex", "AgeCategory", "BMI"],
    },
    {
        "title": "General Health & Well-being",
        "description": "Self-reported health status and symptom duration over the past 30 days.",
        "fields": ["GeneralHealth", "PhysicalHealthDays", "MentalHealthDays", "SleepHours"],
    },
    {
        "title": "Lifestyle & Behaviors",
        "description": "Reported physical activity, smoking, and alcohol intake habits.",
        "fields": ["PhysicalActivity", "Smoking", "AlcoholDrinking"],
    },
    {
        "title": "Medical History",
        "description": "Self-reported previous diagnoses and mobility limitations.",
        "fields": ["Stroke", "Diabetes", "Asthma", "KidneyDisease", "DifficultyWalking"],
    },
]

BRFSS_FIELD_METADATA = {
    "Sex": {
        "label": "Sex",
        "type": "select",
        "help": "Biological sex reported on the survey.",
        "options": {"Female": "Female", "Male": "Male"},
    },
    "AgeCategory": {
        "label": "Age category",
        "type": "select",
        "help": "5-year age group category.",
        "options": {
            "18-24": "18 to 24",
            "25-29": "25 to 29",
            "30-34": "30 to 34",
            "35-39": "35 to 39",
            "40-44": "40 to 44",
            "45-49": "45 to 49",
            "50-54": "50 to 54",
            "55-59": "55 to 59",
            "60-64": "60 to 64",
            "65-69": "65 to 69",
            "70-74": "70 to 74",
            "75-79": "75 to 79",
            "80+": "80 or older",
        },
    },
    "BMI": {
        "label": "BMI",
        "type": "number",
        "unit": "kg/m²",
        "min": 12.0,
        "max": 80.0,
        "step": 0.1,
        "default": 26.5,
        "help": "Body Mass Index calculated as weight (kg) / height (m)².",
    },
    "GeneralHealth": {
        "label": "General health rating",
        "type": "select",
        "help": "Self-rated overall physical and mental health status.",
        "options": {
            "Excellent": "Excellent",
            "Very good": "Very good",
            "Good": "Good",
            "Fair": "Fair",
            "Poor": "Poor",
        },
    },
    "PhysicalHealthDays": {
        "label": "Physical health not good (past 30 days)",
        "type": "number",
        "unit": "days",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 0,
        "help": "Number of days during the past 30 days when physical health was not good.",
    },
    "MentalHealthDays": {
        "label": "Mental health not good (past 30 days)",
        "type": "number",
        "unit": "days",
        "min": 0,
        "max": 30,
        "step": 1,
        "default": 0,
        "help": "Number of days during the past 30 days when stress or depression affected you.",
    },
    "PhysicalActivity": {
        "label": "Physical activity outside work",
        "type": "select",
        "help": "Participated in any physical activities or exercise outside your regular job in past 30 days.",
        "options": {"Yes": "Yes", "No": "No"},
    },
    "SleepHours": {
        "label": "Average sleep hours per day",
        "type": "number",
        "unit": "hours",
        "min": 1,
        "max": 24,
        "step": 1,
        "default": 7,
        "help": "Average hours of sleep per 24-hour period.",
    },
    "Smoking": {
        "label": "Smoked ≥ 100 cigarettes in lifetime",
        "type": "select",
        "help": "CDC standard question: Have you smoked at least 100 cigarettes (5 packs) in your entire life?",
        "options": {"Yes": "Yes", "No": "No"},
    },
    "AlcoholDrinking": {
        "label": "Heavy alcohol consumption",
        "type": "select",
        "help": "Heavy drinking defined by CDC as >14 drinks/week for adult men or >7 drinks/week for women.",
        "options": {"No": "No", "Yes": "Yes"},
    },
    "Stroke": {
        "label": "History of stroke",
        "type": "select",
        "help": "Ever told by a doctor or health professional that you had a stroke.",
        "options": {"No": "No", "Yes": "Yes"},
    },
    "Diabetes": {
        "label": "Diabetes diagnosis",
        "type": "select",
        "help": "Ever told by a doctor that you have diabetes.",
        "options": {
            "No": "No",
            "Yes": "Yes",
            "Borderline": "Borderline / Pre-diabetes",
            "Pregnancy-only": "Yes, during pregnancy only",
        },
    },
    "Asthma": {
        "label": "History of asthma",
        "type": "select",
        "help": "Ever told by a doctor or health professional that you had asthma.",
        "options": {"No": "No", "Yes": "Yes"},
    },
    "KidneyDisease": {
        "label": "History of kidney disease",
        "type": "select",
        "help": "Ever diagnosed with kidney disease (not including kidney stones or bladder infections).",
        "options": {"No": "No", "Yes": "Yes"},
    },
    "DifficultyWalking": {
        "label": "Serious difficulty walking or climbing stairs",
        "type": "select",
        "help": "Do you have serious difficulty walking or climbing stairs?",
        "options": {"No": "No", "Yes": "Yes"},
    },
}

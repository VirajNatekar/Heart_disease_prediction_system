# HeartWise — Heart Disease Prediction System
### Dual-Pipeline Machine Learning Platform: Large-Scale Epidemiological Survey Model & Clinical Benchmark

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://palletsprojects.com/p/flask/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-brightgreen.svg)](tests/)

---

## 1. Executive Summary & Educational Disclaimer

> [!WARNING]
> **Important Non-Diagnostic Notice:** HeartWise is an academic educational machine learning system developed for technical evaluation and pair-programming research. **It is NOT a certified medical device, clinical decision support software, or diagnostic system.** It cannot confirm or exclude cardiovascular disease. Individual patient care decisions must always be made by licensed healthcare professionals. If you or someone you know is experiencing acute cardiac symptoms (chest pain, shortness of breath, pain radiating to the jaw/arm), contact emergency medical services immediately.

HeartWise addresses the dual nature of machine learning in healthcare by implementing **two completely separated model pipelines**:

1. **Model A: Large-Scale CDC Survey Model (CDC BRFSS)**
   - **Data Scale:** 319,795 training respondents (BRFSS 2020) + 246,022 untouched temporal holdout respondents (BRFSS 2022).
   - **Target:** CDC calculated variable `_MICHD` (self-reported history of coronary heart disease or myocardial infarction).
   - **Feature Space:** 15 harmonized demographic, behavioral, and non-target chronic health indicators.
   - **Selected Estimator:** **Histogram Gradient Boosting** with balanced class weighting and optimal decision threshold tuning ($t^* = 0.52$).
   - **Measured Holdout Performance:** ROC-AUC `0.8364`, PR-AUC `0.3382`, Sensitivity `79.64%`, Specificity `72.45%`.

2. **Model B: Clinical Dataset Benchmark (UCI Cleveland)**
   - **Data Scale:** 303 patient records (historical 1989 Cleveland Clinic catheterization cohort).
   - **Target:** Angiographic coronary artery stenosis $\ge$ 50% (`num` mapped to binary 0 vs. 1).
   - **Feature Space:** 13 clinical physiological and imaging attributes (resting BP, serum cholesterol, exercise ST depression, fluoroscopy vessels, thallium scan).
   - **Selected Estimator:** **Extra Trees Classifier** ($N=400$, `min_samples_leaf=2`).
   - **Measured Holdout Performance:** ROC-AUC `0.9448`, Accuracy `86.89%`, Sensitivity `92.86%`, Specificity `81.82%`.

---

## 2. Architectural Separation & Leakage Prevention Invariants

The two models maintain **strict separation across all system layers**:
- **Zero Dataset Concatenation:** Survey respondents and clinical patients are never merged into a single training matrix.
- **Zero Cross-Imputation:** Clinical diagnostic measurements are never imputed into survey records; survey lifestyle questions are never imputed into clinical records.
- **Strict Anti-Leakage Rule:** Survey questions regarding diagnosed heart attack (`HadHeartAttack`, `CVDINFR4`) and angina / coronary heart disease (`HadAngina`, `CVDCRHD4`) are direct components of the calculated outcome `_MICHD`. **These variables are strictly stripped before constructing the feature matrix.**
- **Exclusion of Survey Weights:** Sample weights (`_LLCPWT`) are kept separate from predictive features.
- **Temporal Generalization Test:** BRFSS 2022 is kept untouched during all algorithm selection and threshold tuning to measure genuine real-world generalization across survey years.

---

## 3. Measured Empirical Performance Comparison

### 3.1 Model A (CDC BRFSS Large-Scale Survey Model)
Evaluated on the **2022 Untouched Temporal Holdout** ($N = 246,022$ respondents):

| Metric | Measured Score | Baseline / Context |
|---|---|---|
| **ROC-AUC** | **0.8364** | Chance baseline: 0.5000 |
| **PR-AUC (Average Precision)** | **0.3382** | Population prevalence baseline: 0.0879 (3.8x gain) |
| **Brier Calibration Score** | **0.1755** | Mean squared difference from binary outcome |
| **Sensitivity (Recall)** | **79.64%** | Detected 17,215 of 21,616 reported heart disease cases |
| **Specificity** | **72.45%** | Correctly identified 162,584 of 224,406 non-cases |
| **Balanced Accuracy** | **76.05%** | Macro average of sensitivity and specificity |
| **Precision (PPV)** | **21.78%** | 1 in 4.6 flagged respondents reported heart disease |
| **F1 Score** | **0.3421** | Harmonic mean of precision and recall |

### 3.2 Model B (UCI Cleveland Clinical Benchmark)
Evaluated on **Holdout Test Partition** ($N = 61$ patient records):

| Metric | Measured Score |
|---|---|
| **ROC-AUC** | **0.9448** |
| **Accuracy** | **86.89%** |
| **Balanced Accuracy** | **87.34%** |
| **Sensitivity (Recall)** | **92.86%** |
| **Specificity** | **81.82%** |
| **Precision** | **81.25%** |
| **F1 Score** | **86.67%** |
| **Brier Score** | **0.1044** |

---

## 4. Repository Structure

```text
heart_disease_prediction_system/
├── app.py                         # Flask application & dual-mode routing
├── requirements.txt               # Dependencies
├── README.md                      # Primary project documentation
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
├── artifacts/
│   ├── brfss_model.joblib         # Model A trained pipeline
│   ├── brfss_model_metadata.json  # Model A parameters, thresholds & metrics
│   ├── uci_clinical_model.joblib  # Model B trained pipeline
│   └── uci_clinical_metadata.json # Model B parameters & metrics
├── data/
│   ├── DATASET.md                 # Dataset documentation
│   ├── heart_disease.csv          # Formatted UCI Cleveland clinical dataset
│   └── processed/                 # Partitioned and harmonized CSV files
│       ├── brfss_model/
│       │   ├── brfss_train_2020.csv
│       │   └── brfss_test_2022.csv
│       └── uci_clinical_model/
│           └── uci_clean.csv
├── reports/                       # Evaluation reports and visualizations
├── scripts/
│   └── train_model.py             # Unified model training script
├── src/
│   ├── config.py                  # Dual schemas, metadata & form configs
│   ├── data.py                    # Data ingestion, harmonization & leakage checks
│   ├── modeling.py                # Preprocessing pipelines & candidate algorithms
│   └── validation.py              # Server-side validation for both models
├── static/                        # CSS, JS, and SVG assets
├── templates/                     # Jinja2 HTML templates
└── tests/                         # Pytest test suite (45 unit & integration tests)
```

---

## 5. Execution & Reproduction Guide

### Windows (PowerShell)

```powershell
# 1. Clone repository and navigate to root directory
cd heart_disease_prediction_system

# 2. (Optional) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Train models
python scripts/train_model.py

# 5. Execute the Pytest test suite (45 tests)
python -m pytest

# 6. Launch the web interface
python app.py
```

### macOS / Linux

```bash
cd heart_disease_prediction_system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python -m pytest
python app.py
```

Open `http://127.0.0.1:5000` in any modern web browser.

---

## 6. Web Interface & User Experience

The web application embodies a **minimalist medical design** utilizing white backgrounds, professional deep blue accents (`#1D4ED8`, `#173B70`), crisp typography, and responsive layouts:
- **Mode 1 Tab:** "Survey-Based Heart Disease Status Model" (CDC BRFSS, 15 behavioral & lifestyle attributes).
- **Mode 2 Tab:** "Clinical Dataset Benchmark" (UCI Cleveland, 13 physiological and stress ECG attributes).
- **Result Cards:** Displays predicted class, model likelihood score, tuned decision threshold, model attribution, and explicit non-diagnostic medical disclaimers.
- **Privacy Assurance:** Inputs are evaluated strictly in-memory and never stored to disk or database.

---

## 7. Testing Suite (Pytest)

The test suite contains 45 automated unit and integration tests:
- `tests/test_data.py`: UCI loading, binary mapping, duplicate removal, BRFSS 2020 harmonization, BRFSS 2022 target derivation, and target leakage prevention.
- `tests/test_modeling.py`: Preprocessing pipelines, candidate model generation, threshold optimization algorithms, pipeline fitting, and probability bounds.
- `tests/test_validation.py`: Form validation for both clinical inputs and survey inputs (range checking, type coercions, missing field detection, invalid categorical rejections).
- `tests/test_predictions.py`: End-to-end inference, probability extraction, missing-model handling, feature order invariants.
- `tests/test_routes.py`: Flask routes (`/`, `/health`, `/predict/brfss`, `/predict/clinical`, `/predict`, `404` handler, security headers).

Execute tests via:
```bash
python -m pytest
```

---

## 7. Dataset Provenance & Attribution

- **UCI Heart Disease (Cleveland):** Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease*. American Journal of Cardiology, 64(5), 304–310. [UCI ML Repository](https://archive.ics.uci.edu/dataset/45/heart+disease), DOI: `10.24432/C52P4X`.
- **CDC BRFSS:** Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey Data*. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2020 & 2022.

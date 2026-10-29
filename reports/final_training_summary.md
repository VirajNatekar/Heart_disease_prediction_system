# Final Training & Evaluation Summary: Large-Scale CDC Survey Model

**Date:** 2026-10-22 14:47 UTC  
**Model Name:** Histogram Gradient Boosting  
**Training Cohort:** CDC BRFSS 2020 ($N = 319,795$)  
**Temporal Holdout Cohort:** CDC BRFSS 2022 ($N = 246,022$)  

---

## 1. Executive Summary

A large-scale epidemiological classifier was trained on 319,795 respondents from the 2020 CDC Behavioral Risk Factor Surveillance System (BRFSS) and rigorously evaluated on an untouched temporal holdout of 246,022 respondents from the 2022 BRFSS. 

The selected model (**Histogram Gradient Boosting**) demonstrated strong discrimination and temporal generalization:
- **Temporal Holdout ROC-AUC:** `0.8364`
- **Temporal Holdout PR-AUC:** `0.3382` (3.8x above random chance prevalence of 0.0879)
- **Brier Calibration Score:** `0.1755`
- **Balanced Sensitivity:** `79.6%` at optimal threshold `0.52` (vs. specificity `72.5%`)

---

## 2. Cross-Validation Model Comparison

Candidate algorithms evaluated using 5-fold stratified cross-validation on the 2020 training pool:

| Algorithm Candidate | CV ROC-AUC | CV PR-AUC | Default Threshold Sensitivity | Tuned Threshold Sensitivity | Tuned Specificity |
|---|---|---|---|---|---|
| **Histogram Gradient Boosting** | `0.8423` | `0.3468` | `81.1%` | `79.6%` | `73.5%` |
| **Logistic Regression (Balanced)** | `0.8403` | `0.3439` | `78.2%` | `81.0%` | `72.0%` |
| **Random Forest (Balanced)** | `0.8343` | `0.3335` | `75.3%` | `79.9%` | `72.3%` |

---

## 3. Decision Threshold & Class Imbalance Strategy

In this survey population, respondents with reported heart disease represent approximately **8.6%** of cases. Standard 0.50 cutoff thresholds exhibit poor sensitivity because the majority class dominates probability calibration.

By selecting an optimal decision threshold of **`0.52`** based strictly on training cross-validation Youden's J index, the model achieves a clinical balance:
- **True Positives Detected:** 17,215 out of 21,616 (79.6% sensitivity)
- **True Negatives Identified:** 162,584 out of 224,406 (72.5% specificity)
- **Balanced Accuracy:** `76.0%`

---

## 4. Feature Importance & Epidemiological Associations

The top features identified by the model correspond to established epidemiological risk indicators:

| Rank | Feature | Relative Importance | Epidemiological Context |
|---|---|---|---|
| 1 | **AgeCategory** | `0.0262` | Key self-reported health indicator |
| 2 | **Stroke** | `0.0105` | Key self-reported health indicator |
| 3 | **GeneralHealth** | `0.0053` | Key self-reported health indicator |
| 4 | **PhysicalHealthDays** | `0.0013` | Key self-reported health indicator |
| 5 | **Sex** | `0.0010` | Key self-reported health indicator |
| 6 | **KidneyDisease** | `0.0009` | Key self-reported health indicator |
| 7 | **Smoking** | `0.0007` | Key self-reported health indicator |
| 8 | **MentalHealthDays** | `0.0002` | Key self-reported health indicator |

> [!IMPORTANT]
> **Scientific Interpretation:** These feature importances reflect statistical associations in cross-sectional survey responses. They do **not** establish individual causal mechanisms or guarantee prospective clinical prediction for a patient in a medical clinic.

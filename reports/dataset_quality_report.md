# Dataset Quality & Ingestion Report

**Generated:** 2026-10-22T20:11:53.377635  
**System:** HeartWise Dual-Pipeline Machine Learning System

---

## 1. Dataset Overview & Provenance

| Pipeline / Model | Source Name | Role | Raw Records | Usable Records | Positive Class (%) | Checksum (SHA-256) |
|---|---|---|---|---|---|---|
| **Model B: Clinical** | UCI Cleveland | Clinical Feature Benchmark | 303 | 303 | 45.87% | `953d45066c221ee0...` |
| **Model A: Survey** | CDC BRFSS 2020 | Large-Scale Training Pool | 319,795 | 319,795 | 8.56% | `90a1e2e8bdcb4aba...` |
| **Model A: Survey** | CDC BRFSS 2022 | Untouched Temporal Holdout | 246,022 | 246,022 | 8.79% | `f5eddf47d85170f2...` |

---

## 2. Target Variable Definitions & Class Distributions

### Model B: UCI Cleveland Clinical Benchmark
- **Source Target:** `num` (0: no stenosis; 1, 2, 3, 4: diameter narrowing $\ge$ 50%).
- **Transformation:** Binary mapping ($0 \rightarrow 0$; $1..4 \rightarrow 1$).
- **Class Balance:**
  - Negative ($0$): 164 records (54.1%)
  - Positive ($1$): 139 records (45.87%)

### Model A: CDC BRFSS Survey Model
- **2020 Survey Outcome:** Derived from `HeartDisease` ('Yes' vs 'No') corresponding to CDC calculated variable `_MICHD`.
  - Negative: 292,422 (91.44%)
  - Positive: 27,373 (8.56%)
- **2022 Survey Outcome:** Evaluated as `(HadHeartAttack == 'Yes') | (HadAngina == 'Yes')` adhering strictly to the CDC `_MICHD` codebook definition.
  - Negative: 224,406 (91.21%)
  - Positive: 21,616 (8.79%)

---

## 3. Critical Target Leakage Audit

A comprehensive leakage check was performed across all feature tables:
- **Exclusion of Outcome Components:** Survey questions regarding diagnosed heart attack (`HadHeartAttack`, `CVDINFR4`) and angina / coronary heart disease (`HadAngina`, `CVDCRHD4`) are explicitly stripped before building feature matrices.
- **Exclusion of Survey Weights:** Sample weights (`_LLCPWT`) are not treated as predictive features.
- **Verification Status:** PASSED (Zero target leakage detected).

---

## 4. Missing Data & Handling Strategy

- **UCI Cleveland:** Missing values exist in attributes `ca` (4 records) and `thal` (2 records). Handled strictly inside cross-validation pipelines using median and most-frequent imputation to prevent train-to-test data leakage.
- **CDC BRFSS:** The harmonized cohorts utilize fully validated respondent records across the 15 canonical features. Missing categorical responses were mapped to documented baseline modes, ensuring zero data loss while preserving population fidelity.

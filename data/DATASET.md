# Dataset source and attribution

## Model B: UCI Cleveland Clinical Dataset

**Primary source:** UCI Machine Learning Repository, *Heart Disease* dataset, processed Cleveland subset.

- Dataset page: https://archive.ics.uci.edu/dataset/45/heart+disease
- Processed Cleveland file: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- DOI: https://doi.org/10.24432/C52P4X
- UCI repository license shown on the current dataset page: CC BY 4.0. Follow the repository's attribution requirements if redistributing the downloaded data.
- Underlying paper: Detrano, R. et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." *American Journal of Cardiology*.

UCI describes the source collection as four databases, while the processed Cleveland subset commonly used in teaching contains 303 instances and 13 predictors plus the `num` outcome. The outcome is coded from 0 (no presence) to 4; for this binary project the implementation maps 0 to class 0 and every value greater than 0 to class 1. Missing categorical values are represented as `?` in the original file.

This dataset is small and historical. It is suitable for a reproducible student demonstration, not for establishing general clinical performance. Report the population, test set size and metrics honestly; do not treat the output as a diagnosis.

## Model A: CDC BRFSS Survey Dataset

**Primary source:** Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey Data*.

- Data years: 2020 (training) and 2022 (temporal holdout test)
- Data source: https://www.cdc.gov/brfss/
- License: Public domain (U.S. government data)

The BRFSS is a state-based system of health surveys that collects information on health risk behaviors, chronic health conditions, and use of preventive services. This project uses the calculated variable `_MICHD` (self-reported history of coronary heart disease or myocardial infarction) as the target variable.

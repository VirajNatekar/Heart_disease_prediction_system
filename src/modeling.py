"""Reusable preprocessing pipelines, candidate models, and hyperparameter grids.

Supports:
- Model A: CDC BRFSS Large-Scale Survey Model
- Model B: UCI Cleveland Clinical Benchmark
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    BRFSS_CATEGORICAL_FEATURES,
    BRFSS_NUMERIC_FEATURES,
    UCI_CATEGORICAL_FEATURES,
    UCI_NUMERIC_FEATURES,
)


# ==============================================================================
# MODEL B: UCI CLEVELAND CLINICAL PIPELINE
# ==============================================================================
def build_uci_preprocessor() -> ColumnTransformer:
    """Build preprocessing fitted only within each cross-validation fold."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, UCI_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, UCI_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_uci_candidate_models() -> dict[str, Pipeline]:
    """Candidate algorithms for the 303-row UCI Cleveland clinical dataset."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="liblinear", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "RBF Support Vector Machine": SVC(
            C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.03, max_depth=2, random_state=42
        ),
    }
    return {
        name: Pipeline(
            steps=[("preprocess", build_uci_preprocessor()), ("classifier", classifier)]
        )
        for name, classifier in classifiers.items()
    }


def build_uci_parameter_grids() -> dict[str, dict[str, list]]:
    """Compact bounded hyperparameter grids for UCI tuning."""
    return {
        "Logistic Regression": {"classifier__C": [0.1, 1.0, 10.0]},
        "Random Forest": {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [None, 6],
            "classifier__min_samples_leaf": [1, 3],
        },
        "Extra Trees": {
            "classifier__n_estimators": [150, 300],
            "classifier__max_depth": [None, 6],
            "classifier__min_samples_leaf": [1, 3],
        },
        "RBF Support Vector Machine": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", "auto"],
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [80, 140],
            "classifier__learning_rate": [0.03, 0.08],
            "classifier__max_depth": [1, 2],
        },
    }


# ==============================================================================
# MODEL A: CDC BRFSS LARGE-SCALE SURVEY PIPELINE
# ==============================================================================
def build_brfss_preprocessor() -> ColumnTransformer:
    """Build preprocessor for large-scale BRFSS survey data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, BRFSS_NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, BRFSS_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_brfss_candidate_models() -> dict[str, Pipeline]:
    """Candidate algorithms evaluated on the CDC survey dataset."""
    preprocessor = build_brfss_preprocessor()

    candidates = {
        "Logistic Regression (Balanced)": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Histogram Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=150,
                        learning_rate=0.08,
                        max_leaf_nodes=31,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest (Balanced)": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return candidates


def build_brfss_parameter_grids() -> dict[str, dict[str, list]]:
    """Bounded hyperparameter search grids for large survey data."""
    return {
        "Logistic Regression (Balanced)": {
            "classifier__C": [0.01, 0.1, 1.0],
        },
        "Histogram Gradient Boosting": {
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_leaf_nodes": [20, 31],
        },
        "Random Forest (Balanced)": {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [8, 12],
        },
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    criterion: str = "youden",
) -> Tuple[float, dict]:
    """Find classification threshold optimizing Youden's J or F1 score on training folds."""
    thresholds = np.linspace(0.1, 0.85, 76)
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        tn = np.sum((preds == 0) & (y_true == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

        if criterion == "youden":
            score = sensitivity + specificity - 1.0  # Youden's J statistic
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_threshold = float(t)
            best_metrics = {
                "threshold": float(t),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "precision": float(precision),
                "f1_score": float(f1),
                "youden_j": float(sensitivity + specificity - 1.0),
            }

    return best_threshold, best_metrics


# Backward compatibility aliases
build_preprocessor = build_uci_preprocessor
build_candidate_models = build_uci_candidate_models
build_parameter_grids = build_uci_parameter_grids

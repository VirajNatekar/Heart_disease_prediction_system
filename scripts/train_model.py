"""Train and compare baseline models, then save the selected sklearn pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURE_COLUMNS
from src.data import load_dataset
from src.modeling import build_candidate_models, build_parameter_grids

DATA_PATH = ROOT / "data" / "heart_disease.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "heart_model.joblib"
REPORTS_DIR = ROOT / "reports"
RANDOM_STATE = 42


def _plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.set_title("Hold-out test confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Recorded class")
    ax.set_xticks([0, 1], labels=["No disease recorded", "Disease recorded"])
    ax.set_yticks([0, 1], labels=["No disease recorded", "Disease recorded"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    midpoint = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center",
                    color="white" if matrix[row, column] > midpoint else "black", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curve(y_true: pd.Series, probabilities: np.ndarray, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(fpr, tpr, label=f"Selected model (AUC = {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance reference")
    ax.set_title("Hold-out ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)

    X, y, data_notes = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }
    comparison_rows = []
    tuned_searches = {}
    candidates = build_candidate_models()
    parameter_grids = build_parameter_grids()

    print(f"Records after cleaning/deduplication: {len(X)}")
    print(f"Training records: {len(X_train)} | held-out test records: {len(X_test)}")
    print("Tuning and comparing candidate algorithms with 5-fold CV on training data only...")

    for model_name, pipeline in candidates.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grids[model_name],
            scoring=scorers,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        tuned_searches[model_name] = search
        best_index = int(search.best_index_)
        cv_results = search.cv_results_
        row = {"model": model_name, "best_params": json.dumps(search.best_params_, sort_keys=True)}
        for metric in scorers:
            row[f"cv_{metric}_mean"] = float(cv_results[f"mean_test_{metric}"][best_index])
            row[f"cv_{metric}_std"] = float(cv_results[f"std_test_{metric}"][best_index])
        row["tuned_parameter_combinations"] = int(len(cv_results["params"]))
        comparison_rows.append(row)
        print(
            f"  {model_name:<28} ROC-AUC {row['cv_roc_auc_mean']:.3f} "
            f"(+/- {row['cv_roc_auc_std']:.3f}), recall {row['cv_recall_mean']:.3f}; "
            f"best params {search.best_params_}"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_roc_auc_mean", ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = str(comparison.loc[0, "model"])
    # GridSearchCV's refit=True has already fitted this pipeline on X_train/y_train.
    best_model = tuned_searches[best_name].best_estimator_

    y_pred = best_model.predict(X_test).astype(int)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    metrics = {
        "project": "Heartwise Heart Disease Prediction System",
        "dataset": "UCI Heart Disease — processed Cleveland subset",
        "target_mapping": "num == 0 -> 0; num in 1..4 -> 1",
        "selected_model": best_name,
        "selection_metric": "best mean 5-fold cross-validation ROC-AUC after bounded grid search on training partition",
        "random_state": RANDOM_STATE,
        "split": {"test_size": 0.20, "train_rows": int(len(X_train)), "test_rows": int(len(X_test))},
        "data_notes": data_notes,
        "holdout_test_metrics": {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "precision_positive_class": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall_sensitivity_positive_class": float(recall_score(y_test, y_pred, zero_division=0)),
            "specificity_negative_class": float(tn / (tn + fp)) if (tn + fp) else None,
            "f1_positive_class": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "confusion_matrix_labels_0_1": [[int(v) for v in row] for row in matrix.tolist()],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "classification_report": classification_report(
                y_test, y_pred, labels=[0, 1], target_names=["no disease recorded", "disease recorded"],
                output_dict=True, zero_division=0
            ),
        },
        "limitations": [
            "Small historical dataset; hold-out metrics have substantial uncertainty.",
            "Not externally validated and not a medical diagnosis.",
            "The classifier probability is not calibrated as a patient's clinical risk.",
        ],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (REPORTS_DIR / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    _plot_confusion_matrix(matrix, REPORTS_DIR / "confusion_matrix.png")
    _plot_roc_curve(y_test, probabilities, REPORTS_DIR / "roc_curve.png")

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at_utc": metrics["trained_at_utc"],
        "dataset_name": "UCI Heart Disease — processed Cleveland subset",
        "test_metrics": {
            key: value for key, value in metrics["holdout_test_metrics"].items()
            if key != "classification_report"
        },
        "score_note": "predict_proba output is a raw classifier score and is not a validated clinical risk probability",
    }
    joblib.dump(bundle, ARTIFACT_PATH)

    print("\nSelected model:", best_name)
    print("Hold-out test metrics:")
    for name, value in metrics["holdout_test_metrics"].items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.3f}")
    print(f"\nModel saved to: {ARTIFACT_PATH}")
    print(f"Reports saved to: {REPORTS_DIR}")
    print("Metrics are educational benchmark results only, not clinical validation.")


if __name__ == "__main__":
    main()

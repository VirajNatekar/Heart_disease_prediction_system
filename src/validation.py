"""Input validation for Flask prediction endpoints for both models.

Supports:
- Model A: CDC BRFSS Large-Scale Survey Model
- Model B: UCI Cleveland Clinical Benchmark
"""

from __future__ import annotations

from typing import Any

from src.config import (
    BRFSS_FEATURE_COLUMNS,
    BRFSS_FIELD_METADATA,
    UCI_FEATURE_COLUMNS,
    UCI_FIELD_METADATA,
)


def validate_uci_form(form: Any) -> tuple[dict[str, int | float], list[str]]:
    """Validate and normalize UCI clinical form values before model inference."""
    values: dict[str, int | float] = {}
    errors: list[str] = []

    for field in UCI_FEATURE_COLUMNS:
        meta = UCI_FIELD_METADATA[field]
        raw = str(form.get(field, "")).strip()
        if not raw:
            errors.append(f"{meta['label']} is required.")
            continue

        if meta["type"] == "select":
            options = meta["options"]
            if raw not in options:
                errors.append(f"Select a valid option for {meta['label'].lower()}.")
                continue
            values[field] = int(raw)
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Enter a valid number for {meta['label'].lower()}.")
            continue

        min_val = float(meta["min"])
        max_val = float(meta["max"])
        if number < min_val or number > max_val:
            unit = meta.get("unit", "").strip()
            errors.append(
                f"{meta['label']} must be between {min_val} and {max_val} {unit}.".strip()
            )
            continue

        if float(meta.get("step", 1)) == 1 and not number.is_integer():
            errors.append(f"{meta['label']} must be a whole number.")
            continue

        values[field] = int(number) if number.is_integer() else number

    return values, errors


def validate_brfss_form(form: Any) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize CDC BRFSS survey form values before model inference."""
    values: dict[str, Any] = {}
    errors: list[str] = []

    for field in BRFSS_FEATURE_COLUMNS:
        meta = BRFSS_FIELD_METADATA[field]
        raw = str(form.get(field, "")).strip()
        if not raw:
            errors.append(f"{meta['label']} is required.")
            continue

        if meta["type"] == "select":
            options = meta["options"]
            if raw not in options:
                errors.append(f"Select a valid option for {meta['label'].lower()}.")
                continue
            values[field] = raw
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"Enter a valid number for {meta['label'].lower()}.")
            continue

        min_val = float(meta["min"])
        max_val = float(meta["max"])
        if number < min_val or number > max_val:
            unit = meta.get("unit", "").strip()
            errors.append(
                f"{meta['label']} must be between {min_val} and {max_val} {unit}.".strip()
            )
            continue

        if float(meta.get("step", 1)) == 1 and not number.is_integer():
            errors.append(f"{meta['label']} must be a whole number.")
            continue

        values[field] = int(number) if number.is_integer() else float(number)

    return values, errors


# Backward compatibility alias
validate_form = validate_uci_form

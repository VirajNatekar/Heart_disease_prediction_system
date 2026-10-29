"""Heartwise Flask application. Intended for local academic demonstration.

Dual-Pipeline Architecture:
- Mode 1: Large-Scale CDC Survey Model (CDC BRFSS)
- Mode 2: Clinical Dataset Benchmark (UCI Cleveland)
"""

from __future__ import annotations

import os
from pathlib import Path
import secrets
from typing import Any

import joblib
import pandas as pd
from flask import Flask, render_template, request

from src.config import (
    BRFSS_FEATURE_COLUMNS,
    BRFSS_FIELD_METADATA,
    BRFSS_FORM_SECTIONS,
    BRFSS_MODEL_PATH,
    UCI_FEATURE_COLUMNS,
    UCI_FIELD_METADATA,
    UCI_FORM_SECTIONS,
    UCI_MODEL_PATH,
)
from src.validation import validate_brfss_form, validate_uci_form

ROOT = Path(__file__).resolve().parent
LEGACY_MODEL_PATH = ROOT / "artifacts" / "heart_model.joblib"

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def load_model_bundle(path: Path) -> dict | None:
    """Load a trained model bundle from joblib artifact."""
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            app.logger.error(f"Model artifact at {path} has an unexpected format.")
            return None
        return bundle
    except Exception:
        app.logger.exception(f"Could not load model artifact from {path}.")
        return None


# Primary model bundles
UCI_MODEL_BUNDLE = load_model_bundle(UCI_MODEL_PATH) or load_model_bundle(LEGACY_MODEL_PATH)
BRFSS_MODEL_BUNDLE = load_model_bundle(BRFSS_MODEL_PATH)

# Module-level alias for backward compatibility with existing tests
MODEL_BUNDLE = UCI_MODEL_BUNDLE


@app.after_request
def add_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; img-src 'self' data:; base-uri 'self'; form-action 'self'; frame-ancestors 'none'",
    )
    return response


@app.context_processor
def inject_template_globals():
    return {
        "model_ready": MODEL_BUNDLE is not None,
        "uci_ready": MODEL_BUNDLE is not None,
        "brfss_ready": BRFSS_MODEL_BUNDLE is not None,
        "model_name": MODEL_BUNDLE.get("model_name", "Selected benchmark model") if MODEL_BUNDLE else None,
        "uci_model_name": MODEL_BUNDLE.get("model_name", "Extra Trees Classifier") if MODEL_BUNDLE else "Not Trained",
        "brfss_model_name": BRFSS_MODEL_BUNDLE.get("model_name", "Histogram Gradient Boosting") if BRFSS_MODEL_BUNDLE else "Not Trained",
        "feature_metadata": UCI_FIELD_METADATA,
        "feature_columns": UCI_FEATURE_COLUMNS,
        "uci_sections": UCI_FORM_SECTIONS,
        "uci_metadata": UCI_FIELD_METADATA,
        "uci_features": UCI_FEATURE_COLUMNS,
        "brfss_sections": BRFSS_FORM_SECTIONS,
        "brfss_metadata": BRFSS_FIELD_METADATA,
        "brfss_features": BRFSS_FEATURE_COLUMNS,
    }


def predict(values: dict[str, int | float]) -> dict:
    """Run Model B (UCI Clinical Benchmark) inference."""
    if MODEL_BUNDLE is None:
        raise RuntimeError("The trained model is unavailable. Run scripts/train_model.py first.")

    record = pd.DataFrame([{key: values[key] for key in UCI_FEATURE_COLUMNS}], columns=UCI_FEATURE_COLUMNS)
    model = MODEL_BUNDLE["model"]
    predicted_class = int(model.predict(record)[0])
    raw_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(record)[0]
        classes = list(model.classes_)
        if 1 in classes:
            raw_score = float(probabilities[classes.index(1)])

    return {
        "mode": "clinical",
        "model_type": "Model B: Clinical Feature Benchmark (UCI Cleveland)",
        "model_name": MODEL_BUNDLE.get("model_name", "Selected benchmark model"),
        "positive": predicted_class == 1,
        "raw_score": raw_score,
        "threshold": 0.50,
        "class_label": "Disease Pattern Present (Angiographic >= 50% stenosis)" if predicted_class == 1 else "No Major Narrowing Detected (Angiographic < 50% stenosis)",
        "explanation": (
            "The model flagged an angiographic pattern consistent with diameter narrowing >= 50% in the historical Cleveland dataset. "
            "This is an educational benchmark classification, NOT a medical diagnosis."
            if predicted_class == 1 else
            "The model did not classify this physiological profile in the historical disease group. "
            "This does not rule out cardiovascular disease and must not replace professional clinical evaluation."
        ),
    }


# Primary alias
predict_uci = predict


def predict_brfss(values: dict[str, Any]) -> dict:
    """Run Model A (Large-Scale CDC BRFSS Survey Model) inference."""
    bundle = BRFSS_MODEL_BUNDLE or load_model_bundle(BRFSS_MODEL_PATH)
    if bundle is None:
        raise RuntimeError("The CDC BRFSS survey model is unavailable. Run scripts/train_model.py first.")

    record = pd.DataFrame([{key: values[key] for key in BRFSS_FEATURE_COLUMNS}], columns=BRFSS_FEATURE_COLUMNS)
    model = bundle["model"]
    threshold = float(bundle.get("decision_threshold", 0.52))

    raw_score = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(record)[0]
        classes = list(model.classes_)
        if 1 in classes:
            raw_score = float(probabilities[classes.index(1)])

    if raw_score is not None:
        predicted_class = 1 if raw_score >= threshold else 0
    else:
        predicted_class = int(model.predict(record)[0])

    return {
        "mode": "brfss",
        "model_type": "Model A: Large-Scale CDC Survey Model (CDC BRFSS)",
        "model_name": bundle.get("model_name", "Histogram Gradient Boosting"),
        "positive": predicted_class == 1,
        "raw_score": raw_score,
        "threshold": threshold,
        "class_label": "Reported Heart Disease Pattern Flagged (Survey Association)" if predicted_class == 1 else "Pattern Not Flagged (Typical Non-Reporting Profile)",
        "explanation": (
            f"Based on population survey patterns from 300,000+ CDC BRFSS respondents, the supplied health characteristics align "
            f"with respondents reporting previous coronary heart disease or myocardial infarction (model score: {raw_score*100:.1f}%, threshold: {threshold*100:.1f}%). "
            f"This represents an epidemiological association, NOT a medical diagnosis or individual risk guarantee."
            if predicted_class == 1 else
            f"The reported survey characteristics fall below the tuned classification threshold ({threshold*100:.1f}%) for previously diagnosed coronary heart disease. "
            f"This is not a diagnostic clearance and does not rule out heart disease."
        ),
    }


@app.get("/")
def index():
    mode = request.args.get("mode", "clinical").strip().lower()
    if mode not in ["brfss", "clinical"]:
        mode = "clinical"

    active_sections = BRFSS_FORM_SECTIONS if mode == "brfss" else UCI_FORM_SECTIONS
    active_metadata = BRFSS_FIELD_METADATA if mode == "brfss" else UCI_FIELD_METADATA
    active_features = BRFSS_FEATURE_COLUMNS if mode == "brfss" else UCI_FEATURE_COLUMNS

    return render_template(
        "index.html",
        active_mode=mode,
        sections=active_sections,
        active_metadata=active_metadata,
        active_features=active_features,
        submitted={},
        errors=[],
        result=None,
    )


@app.post("/predict/brfss")
def predict_brfss_route():
    submitted = {field: request.form.get(field, "") for field in BRFSS_FEATURE_COLUMNS}
    values, errors = validate_brfss_form(request.form)
    result = None

    bundle = BRFSS_MODEL_BUNDLE or load_model_bundle(BRFSS_MODEL_PATH)
    if not errors:
        if bundle is None:
            errors.append("The CDC BRFSS survey model is not trained yet. Run `python scripts/train_model.py`.")
        else:
            try:
                result = predict_brfss(values)
            except Exception:
                app.logger.exception("BRFSS Prediction failed")
                errors.append("The survey prediction could not be completed. Check model artifacts and restart.")

    return render_template(
        "index.html",
        active_mode="brfss",
        sections=BRFSS_FORM_SECTIONS,
        active_metadata=BRFSS_FIELD_METADATA,
        active_features=BRFSS_FEATURE_COLUMNS,
        submitted=submitted,
        errors=errors,
        result=result,
    ), (400 if errors else 200)


@app.post("/predict/clinical")
def predict_clinical_route():
    submitted = {field: request.form.get(field, "") for field in UCI_FEATURE_COLUMNS}
    values, errors = validate_uci_form(request.form)
    result = None

    if not errors:
        if MODEL_BUNDLE is None:
            errors.append("The prediction model is not trained yet. Follow the setup steps in README.md, then refresh this page.")
        else:
            try:
                result = predict(values)
            except Exception:
                app.logger.exception("Prediction failed")
                errors.append("The prediction could not be completed. Check the model artifact and restart the application.")

    return render_template(
        "index.html",
        active_mode="clinical",
        sections=UCI_FORM_SECTIONS,
        active_metadata=UCI_FIELD_METADATA,
        active_features=UCI_FEATURE_COLUMNS,
        submitted=submitted,
        errors=errors,
        result=result,
    ), (400 if errors else 200)


@app.post("/predict")
def predict_route():
    """Unified route with backward compatibility."""
    form_keys = set(request.form.keys())
    if "mode" in request.form and request.form.get("mode") == "brfss":
        return predict_brfss_route()
    if any(k in BRFSS_FEATURE_COLUMNS for k in form_keys) and not any(k in ["cp", "trestbps", "thalach"] for k in form_keys):
        return predict_brfss_route()
    return predict_clinical_route()


@app.get("/health")
def health():
    """Health check endpoint reporting readiness of both model pipelines."""
    uci_status = MODEL_BUNDLE is not None
    brfss_status = (BRFSS_MODEL_BUNDLE is not None) or BRFSS_MODEL_PATH.exists()
    return {
        "status": "ok",
        "model_ready": uci_status,
        "models": {
            "uci_clinical": uci_status,
            "cdc_brfss": brfss_status,
        },
    }, (200 if uci_status else 503)


@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Internal server error")
    return render_template("500.html"), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0").strip() == "1"
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=debug_mode,
    )

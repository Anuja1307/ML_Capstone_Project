"""
app/predict_subprocess.py

Standalone inference worker. Run ONLY with Anaconda Python.
Called by pipeline_utils.py via subprocess.
Reads a JSON payload from stdin, loads the model, runs inference,
and writes a JSON result to stdout.

NOTE: This module does NOT import from pipeline_utils.py.
      It contains the inference logic directly (matching the notebook pipelines).
"""

import sys
import json
import os
from pathlib import Path

# Navigate to project root (parent of app/)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib


# ── Regression inference (matches notebooks/regression.ipynb) ─────────────

def _predict_regression(inputs: dict) -> dict:
    model = joblib.load("models/regression_model.pkl")
    with open("models/regression_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Build full feature record from training defaults
    record = metadata["feature_defaults"].copy()
    record.update(inputs)

    # Feature engineering — matches regression.ipynb
    record["TotalSF"] = (
        float(record.get("TotalBsmtSF", 0))
        + float(record.get("1stFlrSF", 0))
        + float(record.get("2ndFlrSF", 0))
    )
    record["HouseAge"] = int(record.get("YrSold", 2008)) - int(record.get("YearBuilt", 1973))
    record["TotalBath"] = (
        float(record.get("FullBath", 2))
        + 0.5 * float(record.get("HalfBath", 0))
        + float(record.get("BsmtFullBath", 0))
        + 0.5 * float(record.get("BsmtHalfBath", 0))
    )

    # LotFrontage: use neighborhood median if not supplied
    if "LotFrontage" not in inputs or inputs.get("LotFrontage") is None:
        neigh = record.get("Neighborhood", "NAmes")
        record["LotFrontage"] = metadata.get("neighborhood_medians", {}).get(
            neigh, metadata.get("global_lot_median", 69.0)
        )

    # Build input DataFrame in training feature order
    all_features = metadata["all_features"]
    input_df = pd.DataFrame([{col: record.get(col, 0) for col in all_features}])

    # Predict log-price → invert via expm1
    log_pred = model.predict(input_df)[0]
    predicted_price = float(max(10000.0, np.expm1(log_pred)))
    return {"predicted_price": predicted_price}


# ── Classification inference (matches notebooks/Classification.ipynb) ──────

def _predict_classification(inputs: dict, model_key: str) -> dict:
    model = joblib.load(f"models/classification_{model_key}.pkl")
    with open("models/classification_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    record = inputs.copy()

    # Fallback TotalCharges
    monthly = float(record.get("MonthlyCharges", 50.0))
    tenure = int(record.get("tenure", 1))
    if not record.get("TotalCharges") or float(record.get("TotalCharges", 0)) <= 0:
        record["TotalCharges"] = round(tenure * monthly, 2)

    # TotalServices feature engineering (matches notebook Cell 53)
    service_cols = metadata.get("service_cols", [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ])
    record["TotalServices"] = sum(1 for col in service_cols if record.get(col) == "Yes")

    input_df = pd.DataFrame([record])

    # Apply training-derived IQR outlier bounds
    for col, bounds in metadata.get("outlier_bounds", {}).items():
        if col in input_df.columns and len(bounds) == 2:
            input_df[col] = input_df[col].clip(lower=bounds[0], upper=bounds[1])

    pred_class = int(model.predict(input_df)[0])

    # Probability (only when model supports it)
    prob_churn = None
    estimator = model.named_steps.get("model")
    if hasattr(estimator, "predict_proba"):
        try:
            proba = model.predict_proba(input_df)[0]
            prob_churn = float(proba[1])
        except Exception:
            prob_churn = None

    return {"pred_class": pred_class, "prob_churn": prob_churn}


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {exc}"}))
        sys.exit(1)

    task = payload.get("task", "")
    try:
        if task == "regression":
            result = _predict_regression(payload["inputs"])
        elif task == "classification":
            result = _predict_classification(payload["inputs"], payload["model_key"])
        else:
            raise ValueError(f"Unknown task: {task!r}")
        print(json.dumps({"success": True, "result": result}))
    except Exception as exc:
        import traceback as _tb
        print(json.dumps({
            "success": False,
            "error": str(exc),
            "traceback": _tb.format_exc(),
        }))
        sys.exit(1)


if __name__ == "__main__":
    main()

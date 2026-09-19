"""Direct inference helpers for the Streamlit application.

The fitted scikit-learn pipelines in ``models/`` include preprocessing and the
estimator. They are loaded in-process with Joblib; prediction never starts a
second Python interpreter.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def _load_model(model_filename: str):
    """Load and cache one fitted pipeline from the repository's models folder."""
    model_path = MODEL_DIR / model_filename
    if not model_path.is_file():
        raise FileNotFoundError(
            "Model file not found. Please generate the model artifacts first."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model '{model_filename}': {exc}") from exc


@st.cache_data
def _load_metadata(metadata_filename: str) -> dict:
    metadata_path = MODEL_DIR / metadata_filename
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Model file not found. Please generate the model artifacts first."
        )
    try:
        with metadata_path.open(encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model metadata '{metadata_filename}': {exc}") from exc


def predict_regression_price(user_inputs: Dict[str, Any]) -> float:
    """Predict Ames sale price using the serialized Lasso pipeline."""
    model = _load_model("regression_model.pkl")
    metadata = _load_metadata("regression_metadata.json")

    record = metadata["feature_defaults"].copy()
    record.update(user_inputs)
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

    if "LotFrontage" not in user_inputs or user_inputs.get("LotFrontage") is None:
        neighborhood = record.get("Neighborhood", "NAmes")
        record["LotFrontage"] = metadata.get("neighborhood_medians", {}).get(
            neighborhood, metadata.get("global_lot_median", 69.0)
        )

    features = metadata["all_features"]
    input_df = pd.DataFrame([{column: record.get(column, 0) for column in features}])
    log_prediction = model.predict(input_df)[0]
    return float(max(10000.0, np.expm1(log_prediction)))


def predict_classification_churn(
    user_inputs: Dict[str, Any], model_key: str
) -> Tuple[int, Optional[float]]:
    """Predict Telco churn with a selected serialized classification pipeline."""
    model = _load_model(f"classification_{model_key}.pkl")
    metadata = _load_metadata("classification_metadata.json")
    record = user_inputs.copy()

    monthly_charges = float(record.get("MonthlyCharges", 50.0))
    tenure = int(record.get("tenure", 1))
    if not record.get("TotalCharges") or float(record.get("TotalCharges", 0)) <= 0:
        record["TotalCharges"] = round(tenure * monthly_charges, 2)

    service_columns = metadata.get("service_cols", [])
    record["TotalServices"] = sum(
        1 for column in service_columns if record.get(column) == "Yes"
    )
    input_df = pd.DataFrame([record])
    for column, bounds in metadata.get("outlier_bounds", {}).items():
        if column in input_df.columns and len(bounds) == 2:
            input_df[column] = input_df[column].clip(lower=bounds[0], upper=bounds[1])

    predicted_class = int(model.predict(input_df)[0])
    probability = None
    estimator = model.named_steps.get("model")
    if hasattr(estimator, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    return predicted_class, probability

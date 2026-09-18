"""
app/model_loader.py

Metadata-only loader for the ML Capstone GUI.

Loads only JSON files (no sklearn / joblib dependency).
The serialised .pkl pipelines are loaded inside predict_subprocess.py
which runs under Anaconda Python — keeping the Streamlit process free
of any sklearn C-extension imports.
"""

import json
import traceback
from pathlib import Path
import streamlit as st

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load_regression_metadata():
    """Load regression metadata JSON (feature schema, metrics, defaults)."""
    meta_path = MODELS_DIR / "regression_metadata.json"
    if not meta_path.exists():
        st.error(f"Regression metadata not found at {meta_path}.")
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        traceback.print_exc()
        st.error(f"Failed to load regression metadata: {exc}")
        return None


@st.cache_resource
def load_classification_metadata():
    """Load classification metadata JSON (IQR bounds, feature lists, metrics)."""
    meta_path = MODELS_DIR / "classification_metadata.json"
    if not meta_path.exists():
        st.error(f"Classification metadata not found at {meta_path}.")
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        traceback.print_exc()
        st.error(f"Failed to load classification metadata: {exc}")
        return None

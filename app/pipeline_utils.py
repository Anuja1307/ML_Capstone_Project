"""
app/pipeline_utils.py

Inference wrappers for the ML Capstone GUI.

All model inference is delegated to predict_subprocess.py running under
Anaconda Python, so this module is safe to import from ANY Python environment
(including Python 3.12 whose sklearn DLLs are blocked by Application Control).
No sklearn/joblib imports occur in the Streamlit process.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Anaconda Python path — models were serialised with this interpreter
ANACONDA_PYTHON = r"C:\ProgramData\anaconda3\python.exe"

_APP_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = _APP_DIR / "predict_subprocess.py"
PROJECT_ROOT = _APP_DIR.parent

SUBPROCESS_TIMEOUT = 60  # seconds


def _run_predict_subprocess(payload: dict) -> dict:
    """
    Serialise *payload* as JSON, pipe it to predict_subprocess.py running under
    Anaconda Python, and return the parsed JSON result dict.
    Raises RuntimeError with a detailed message on any failure.
    """
    # Verify Anaconda Python exists before trying to spawn
    if not Path(ANACONDA_PYTHON).exists():
        raise RuntimeError(
            f"Anaconda Python not found at:\n  {ANACONDA_PYTHON}\n\n"
            "Please verify that Anaconda is installed at C:\\ProgramData\\anaconda3\\"
        )

    try:
        proc = subprocess.run(
            [ANACONDA_PYTHON, str(PREDICT_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=SUBPROCESS_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Prediction subprocess timed out after {SUBPROCESS_TIMEOUT}s."
        )

    # Log stderr to the Streamlit terminal for debugging
    if proc.stderr.strip():
        print("[predict_subprocess stderr]", proc.stderr, file=sys.stderr)

    stdout = proc.stdout.strip()

    if not stdout:
        raise RuntimeError(
            f"Prediction subprocess produced no output (exit code {proc.returncode}).\n"
            f"Stderr:\n{proc.stderr}"
        )

    try:
        response = json.loads(stdout)
    except json.JSONDecodeError:
        raise RuntimeError(
            f"Prediction subprocess returned invalid JSON (exit {proc.returncode}):\n"
            f"{stdout}\nStderr:\n{proc.stderr}"
        )

    if not response.get("success"):
        tb = response.get("traceback", "")
        print("[predict_subprocess traceback]\n", tb, file=sys.stderr)
        raise RuntimeError(
            f"Prediction failed: {response.get('error', 'unknown error')}"
        )

    return response["result"]


def predict_regression_price(user_inputs: Dict[str, Any]) -> float:
    """
    Run Ames Housing Lasso regression via Anaconda Python subprocess.
    Returns the predicted SalePrice in USD.
    """
    payload = {"task": "regression", "inputs": user_inputs}
    result = _run_predict_subprocess(payload)
    return float(result["predicted_price"])


def predict_classification_churn(
    user_inputs: Dict[str, Any],
    model_key: str,
) -> Tuple[int, Optional[float]]:
    """
    Run Telco churn classification via Anaconda Python subprocess.
    Returns (pred_class, prob_churn) where prob_churn is None for SVC.
    """
    payload = {
        "task": "classification",
        "inputs": user_inputs,
        "model_key": model_key,
    }
    result = _run_predict_subprocess(payload)
    return int(result["pred_class"]), result.get("prob_churn")

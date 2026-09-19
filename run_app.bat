@echo off
REM ============================================================
REM  run_app.bat — Launch ML Capstone Streamlit GUI
REM  Uses the Python environment from which this script is launched
REM ============================================================

echo Starting ML Capstone Prediction GUI...
echo Using the active Python environment
echo.
echo Once started, open: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

python -m streamlit run app/app.py

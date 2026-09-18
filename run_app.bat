@echo off
REM ============================================================
REM  run_app.bat — Launch ML Capstone Streamlit GUI
REM  Uses Anaconda Python (required: models were built with it)
REM ============================================================

echo Starting ML Capstone Prediction GUI...
echo Using Anaconda Python (required for sklearn/scipy model compatibility)
echo.
echo Once started, open: http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

"C:\ProgramData\anaconda3\Scripts\streamlit.exe" run app/app.py

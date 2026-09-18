# ============================================================
#  run_app.ps1 — Launch ML Capstone Streamlit GUI (PowerShell)
#  Uses Anaconda Python (required: models were built with it)
# ============================================================

Write-Host "Starting ML Capstone Prediction GUI..." -ForegroundColor Cyan
Write-Host "Using Anaconda Python (required for sklearn/scipy model compatibility)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Once started, open: http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

& "C:\ProgramData\anaconda3\Scripts\streamlit.exe" run app/app.py

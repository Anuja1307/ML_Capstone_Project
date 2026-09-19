# ============================================================
#  run_app.ps1 — Launch ML Capstone Streamlit GUI (PowerShell)
#  Uses the Python environment from which this script is launched
# ============================================================

Write-Host "Starting ML Capstone Prediction GUI..." -ForegroundColor Cyan
Write-Host "Using the active Python environment" -ForegroundColor Yellow
Write-Host ""
Write-Host "Once started, open: http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server."
Write-Host ""

python -m streamlit run app/app.py

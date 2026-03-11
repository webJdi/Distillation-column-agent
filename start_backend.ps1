# CDU Optimizer — Run Scripts
# -----------------------------------------------------------
# Start the FastAPI backend
# -----------------------------------------------------------

Write-Host "Starting CDU Optimizer Backend..." -ForegroundColor Cyan
Write-Host "  API docs: http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host ""

# Activate venv if it exists
$venvPath = Join-Path $PSScriptRoot "distvenv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "  Virtual environment activated" -ForegroundColor Green
}

# Run FastAPI with uvicorn
Set-Location $PSScriptRoot
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

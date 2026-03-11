# CDU Optimizer — Start Frontend
# -----------------------------------------------------------

Write-Host "Starting CDU Optimizer Frontend..." -ForegroundColor Cyan
Write-Host "  App: http://localhost:5173" -ForegroundColor Gray
Write-Host ""

Set-Location (Join-Path $PSScriptRoot "frontend")

# Install deps if needed
if (-not (Test-Path "node_modules")) {
    Write-Host "  Installing npm dependencies..." -ForegroundColor Yellow
    npm install
}

npm run dev

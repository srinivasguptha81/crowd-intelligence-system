# CrowdGuard AI - Backend Launcher for Windows PowerShell
# Resolves a working Python virtual environment (.venv or venv) automatically.

Set-Location $PSScriptRoot

$Candidates = @(
    (Join-Path $PSScriptRoot "venv\Scripts\python.exe"),
    (Join-Path $PSScriptRoot ".venv\Scripts\python.exe")
)

$VenvPython = $null
foreach ($Cand in $Candidates) {
    if (Test-Path $Cand) {
        try {
            & $Cand -c "import fastapi, uvicorn" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $VenvPython = $Cand
                break
            }
        } catch {}
    }
}

if (-not $VenvPython) {
    Write-Host "No working virtual environment found. Running setup.ps1..." -ForegroundColor Yellow
    powershell -ExecutionPolicy Bypass -File .\setup.ps1
    $VenvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
}

Write-Host "Using Python: $VenvPython" -ForegroundColor DarkGray
Write-Host "Starting CrowdGuard AI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
& $VenvPython -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# CrowdGuard AI - Backend Setup Script for Windows (PowerShell)
# This script sets up a reliable Python 3.11 virtual environment and installs dependencies.

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host " CrowdGuard AI - Backend Setup (Windows / Python 3.11)" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# 1. Check for Python 3.10 - 3.12
Write-Host "`n[1/6] Checking Python 3.10-3.12 availability..." -ForegroundColor Yellow
$PythonCmd = $null

foreach ($minor in @("3.11", "3.12", "3.10")) {
    if ($PythonCmd) { break }
    try {
        $ver = & py "-$minor" --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $ver -match "3\.1[0-2]") {
            $PythonCmd = @("py", "-$minor")
            Write-Host "  Found via Python launcher: $ver" -ForegroundColor Green
        }
    } catch {}
}

# Fallback to python.exe on PATH
if (-not $PythonCmd) {
    try {
        $ver = & python --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $ver -match "3\.1[0-2]") {
            $PythonCmd = @("python")
            Write-Host "  Found via PATH: $ver" -ForegroundColor Green
        }
    } catch {}
}

if (-not $PythonCmd) {
    Write-Host "ERROR: Python 3.10-3.12 was not detected on this system." -ForegroundColor Red
    Write-Host "Please install Python 3.11 or 3.12 and ensure 'Add python.exe to PATH' is checked." -ForegroundColor Red
    exit 1
}

# 2. Check / Create Virtual Environment
Write-Host "`n[2/6] Verifying virtual environment (venv)..." -ForegroundColor Yellow
$VenvDir = Join-Path $ScriptDir "venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

$NeedsCreate = -not (Test-Path $VenvPython)
if (-not $NeedsCreate) {
    try {
        & $VenvPython -c "import sys" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Existing venv points to an invalid Python interpreter; recreating..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $VenvDir
            $NeedsCreate = $true
        }
    } catch {
        $NeedsCreate = $true
    }
}

if ($NeedsCreate) {
    Write-Host "  Creating new virtual environment at $VenvDir..." -ForegroundColor Cyan
    if ($PythonCmd.Count -gt 1) {
        & $PythonCmd[0] $PythonCmd[1] -m venv $VenvDir
    } else {
        & $PythonCmd[0] -m venv $VenvDir
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $VenvPython)) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "  Existing virtual environment found at $VenvDir." -ForegroundColor Green
}

# 3. Upgrade pip
Write-Host "`n[3/6] Upgrading pip..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --default-timeout 100
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Pip upgrade encountered a warning, continuing..." -ForegroundColor Yellow
}

# 4. Clean up any conflicting opencv packages
Write-Host "`n[4/6] Checking for OpenCV package conflicts..." -ForegroundColor Yellow
try {
    $hasHeadless = & $VenvPython -m pip list | Select-String "opencv-python-headless"
    $hasStandard = & $VenvPython -m pip list | Select-String "opencv-python "
    if ($hasHeadless -and $hasStandard) {
        Write-Host "  Detected conflicting opencv-python and opencv-python-headless." -ForegroundColor Yellow
        Write-Host "  Removing opencv-python-headless to preserve clean opencv-python installation..." -ForegroundColor Cyan
        & $VenvPython -m pip uninstall -y opencv-python-headless
    }
} catch {}

# 5. Install dependencies from requirements.txt with timeout and retries
Write-Host "`n[5/6] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "  Note: PyTorch CPU and OpenCV wheels may take a moment to download." -ForegroundColor DarkGray
Write-Host "  Using --default-timeout 120 --retries 5 for reliable networking." -ForegroundColor DarkGray

& $VenvPython -m pip install --default-timeout 120 --retries 5 -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: pip install failed with exit code $LASTEXITCODE." -ForegroundColor Red
    Write-Host "Recovery options:" -ForegroundColor Yellow
    Write-Host "  1. If a timeout occurred, check your network and re-run: .\setup.ps1" -ForegroundColor Yellow
    Write-Host "  2. Or install PyTorch CPU explicitly:" -ForegroundColor Yellow
    Write-Host "     .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" -ForegroundColor Yellow
    Write-Host "  3. Then install remaining requirements:" -ForegroundColor Yellow
    Write-Host "     .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 6. Verify imports and configuration
Write-Host "`n[6/6] Verifying installed packages and backend imports..." -ForegroundColor Yellow

# Ensure .env exists
$EnvFile = Join-Path $ScriptDir ".env"
$EnvExample = Join-Path $ScriptDir ".env.example"
if (-not (Test-Path $EnvFile) -and (Test-Path $EnvExample)) {
    Write-Host "  Creating backend .env from .env.example..." -ForegroundColor Cyan
    Copy-Item $EnvExample $EnvFile
}

$VerifyScript = @'
import sys
required_modules = [
    ('fastapi', 'FastAPI'),
    ('uvicorn', 'Uvicorn'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('torch', 'PyTorch'),
    ('torchvision', 'Torchvision'),
    ('ultralytics', 'Ultralytics YOLO'),
    ('sklearn', 'Scikit-learn'),
    ('joblib', 'Joblib'),
    ('app.main', 'CrowdGuard Main App')
]

failed = []
for mod, name in required_modules:
    try:
        __import__(mod)
        print(f"  [OK] {name} imported successfully.")
    except Exception as exc:
        print(f"  [FAIL] {name} ({mod}): {exc}")
        failed.append(name)

if failed:
    sys.exit(1)
print("  All core dependencies verified successfully!")
'@

$VerifyScript | & $VenvPython -
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nERROR: One or more required packages failed verification." -ForegroundColor Red
    exit 1
}

Write-Host "`n=====================================================" -ForegroundColor Green
Write-Host " Backend setup complete and fully verified!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "To start the backend, run:" -ForegroundColor Cyan
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  .\venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000" -ForegroundColor White
Write-Host "Endpoints:" -ForegroundColor Cyan
Write-Host "  - Health: http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host "  - Docs:   http://127.0.0.1:8000/docs" -ForegroundColor White

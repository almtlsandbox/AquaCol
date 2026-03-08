# build.ps1 — Build AquaCol as a standalone Windows executable
# Usage:  .\build.ps1
# Output: dist\AquaCol.exe

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── 1. Ensure PyInstaller is installed ────────────────────────────────────────
Write-Host "`n[1/4] Checking PyInstaller..." -ForegroundColor Cyan
python -m pip install --quiet --upgrade pyinstaller
if ($LASTEXITCODE -ne 0) { throw "pip install pyinstaller failed" }

# ── 2. Clean previous build artefacts ────────────────────────────────────────
Write-Host "[2/4] Cleaning previous build..." -ForegroundColor Cyan
foreach ($dir in @("build", "dist")) {
    if (Test-Path $dir) {
        Remove-Item $dir -Recurse -Force
        Write-Host "  Removed $dir\"
    }
}
if (Test-Path "AquaCol.spec") {
    Remove-Item "AquaCol.spec" -Force
    Write-Host "  Removed AquaCol.spec"
}

# ── 3. Run PyInstaller ────────────────────────────────────────────────────────
Write-Host "[3/4] Building executable..." -ForegroundColor Cyan
python -m PyInstaller `
    --onefile `
    --windowed `
    --name AquaCol `
    --hidden-import cv2 `
    --collect-all cv2 `
    --hidden-import scipy.ndimage `
    --hidden-import PIL._tkinter_finder `
    underwater_enhancement_gui.py

if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }

# ── 4. Report ─────────────────────────────────────────────────────────────────
Write-Host "[4/4] Done." -ForegroundColor Green
$exe = "dist\AquaCol.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "`n  Output : $((Resolve-Path $exe).Path)" -ForegroundColor Yellow
    Write-Host "  Size   : $size MB" -ForegroundColor Yellow
} else {
    throw "Expected output not found: $exe"
}

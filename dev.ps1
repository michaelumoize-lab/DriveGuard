# DriveGuard Full-Stack Dev Launcher for PowerShell
param(
    [ValidateSet("all", "backend", "frontend")]
    [string]$Target = "all"
)

$root = $PSScriptRoot

if ($Target -eq "backend" -or $Target -eq "all") {
    Write-Host "[DriveGuard] Launching FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\.venv311\Scripts\activate.ps1; python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"
}

if ($Target -eq "frontend" -or $Target -eq "all") {
    Write-Host "[DriveGuard] Launching Next.js Frontend on http://localhost:3000 ..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root\frontend'; npm run dev"
}

Write-Host "Services started! Frontend: http://localhost:3000 | Backend: http://127.0.0.1:8000" -ForegroundColor Yellow

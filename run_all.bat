@echo off
title DriveGuard - Full Stack Launcher

:: Automatically resolve Project Root whether run from root or from launchers/
set "ROOT=%~dp0"
if not exist "%ROOT%frontend\package.json" (
    set "ROOT=%~dp0..\"
)
cd /d "%ROOT%"

echo ============================================================
echo   DriveGuard - Launching Full Stack (Backend + Frontend)
echo   Project Root: %CD%
echo ============================================================
echo.

echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "DriveGuard Backend (FastAPI)" cmd /k "cd /d ""%CD%"" && set LOCAL_CAMERA_API_ENABLED=true && call .\.venv311\Scripts\activate.bat && python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting Next.js Frontend on http://localhost:3000 ...
start "DriveGuard Frontend (Next.js)" cmd /k "cd /d ""%CD%\frontend"" && npm run dev"

echo.
echo ============================================================
echo   Both services are starting in dedicated windows!
echo   - Backend:   http://127.0.0.1:8000  (API Docs at /docs)
echo   - Frontend:  http://localhost:3000
echo ============================================================

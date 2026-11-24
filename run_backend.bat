@echo off
title DriveGuard - Python Backend (FastAPI)

:: Automatically resolve Project Root whether run from root or from launchers/
set "ROOT=%~dp0"
if not exist "%ROOT%frontend\package.json" (
    set "ROOT=%~dp0..\"
)
cd /d "%ROOT%"

echo ============================================================
echo   DriveGuard - Starting Local Python AI Backend
echo   Project Root: %CD%
echo   Host: http://127.0.0.1:8000
echo   Docs: http://127.0.0.1:8000/docs
echo ============================================================
echo.

set LOCAL_CAMERA_API_ENABLED=true
call .\.venv311\Scripts\activate.bat
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

pause

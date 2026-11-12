@echo off
title DriveGuard - Automated Test Suite
cd /d "%~dp0\.."

echo ============================================================
echo   DriveGuard - Running Automated Test Suite (pytest)
echo ============================================================
echo.

".\.venv311\Scripts\pytest.exe" -v

echo.
pause

@echo off
title DriveGuard - Laptop Built-in Webcam
cd /d "%~dp0\.."

echo ============================================================
echo   DriveGuard - Launching Laptop Built-in Webcam (Phase 1)
echo   Device Index: 0 (DirectShow)
echo   Tip: Press 'm' to toggle mirror on/off
echo        Press 'q' or ESC to exit
echo ============================================================
echo.

".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --device 0 %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Camera runner exited with error code %ERRORLEVEL%.
    pause
)

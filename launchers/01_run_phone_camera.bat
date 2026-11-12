@echo off
title DriveGuard - Phone Camera Stream
cd /d "%~dp0\.."

echo ============================================================
echo   DriveGuard - Launching Phone Camera Stream (Phase 2)
echo   Target: http://192.168.0.103:8080/video
echo   Tip: Press 'm' in preview window to toggle mirror on/off
echo        Press 'q' or ESC to exit
echo ============================================================
echo.

".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url http://192.168.0.103:8080/video %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Camera runner exited with error code %ERRORLEVEL%.
    pause
)

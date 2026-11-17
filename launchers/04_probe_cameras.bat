@echo off
title DriveGuard - Probe Local Cameras
cd /d "%~dp0\.."

echo ============================================================
echo   DriveGuard - Probing Available Video Capture Devices
echo ============================================================
echo.

".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --probe

echo.
pause

@echo off
title DriveGuard - Automated Test Suite
cd /d "%~dp0\.."

echo ============================================================
echo   DriveGuard - Running Automated Test Suite (pytest)
echo ============================================================
echo.

".\.venv311\Scripts\pytest.exe" backend/tests/test_camera_subsystem.py backend/tests/test_frame_quality.py backend/tests/test_phone_camera.py backend/tests/test_face_landmarker.py -v

echo.
pause

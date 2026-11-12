@echo off
setlocal enabledelayedexpansion
title DriveGuard - Phone Camera Stream
cd /d "%~dp0\.."

set "DEFAULT_URL=http://192.168.0.103:8080/video"
set "PHONE_URL="

:: 1. CLI argument override
if not "%~1"=="" (
    set "PHONE_URL=%~1"
    shift
    goto :RUN_STREAM
)

:: 2. Environment variable override
if not "%DRIVEGUARD_PHONE_URL%"=="" (
    set "PHONE_URL=%DRIVEGUARD_PHONE_URL%"
    goto :RUN_STREAM
)

:: 3. Read configured URL from launchers\phone_url.txt
if exist "%~dp0phone_url.txt" (
    set /p PHONE_URL=<"%~dp0phone_url.txt"
)

:: 4. Fallback if empty
if "%PHONE_URL%"=="" set "PHONE_URL=%DEFAULT_URL%"

echo ============================================================
echo   DriveGuard - Launching Phone Camera Stream (Phase 2)
echo ============================================================
echo Configured stream URL: %PHONE_URL%
echo.
echo Press ENTER to use configured URL, or type new URL below:
set /p USER_INPUT="URL [%PHONE_URL%]: "
if not "!USER_INPUT!"=="" (
    set "PHONE_URL=!USER_INPUT!"
    echo !USER_INPUT!> "%~dp0phone_url.txt"
)

:RUN_STREAM
echo.
echo Connecting to stream: %PHONE_URL%
echo Tip: Press 'm' in preview window to toggle mirror on/off
echo      Press 'q' or ESC to exit
echo ============================================================
echo.

".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url %PHONE_URL% %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Camera runner exited with error code %ERRORLEVEL%.
    pause
)

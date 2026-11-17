@echo off
setlocal enabledelayedexpansion
title DriveGuard - Phone Camera Stream
cd /d "%~dp0\.."

set "URL_1=http://192.168.0.135:4747/video"
set "URL_2=http://192.168.0.103:8080/video"
set "PHONE_URL="
set "EXTRA_ARGS="

:: 1. CLI argument override
if "%~1"=="" goto :PROMPT_MENU

set "FIRST_ARG=%~1"
if "!FIRST_ARG!"=="1" (
    set "PHONE_URL=%URL_1%"
    shift
    goto :COLLECT_EXTRA_ARGS
)
if "!FIRST_ARG!"=="2" (
    set "PHONE_URL=%URL_2%"
    shift
    goto :COLLECT_EXTRA_ARGS
)
if "!FIRST_ARG:~0,4!"=="http" (
    set "PHONE_URL=%~1"
    shift
    goto :COLLECT_EXTRA_ARGS
)
if "!FIRST_ARG:~0,4!"=="rtsp" (
    set "PHONE_URL=%~1"
    shift
    goto :COLLECT_EXTRA_ARGS
)

:COLLECT_EXTRA_ARGS
if not "%~1"=="" (
    set "EXTRA_ARGS=!EXTRA_ARGS! %1"
    shift
    goto :COLLECT_EXTRA_ARGS
)

if not "!PHONE_URL!"=="" goto :RUN_STREAM

:PROMPT_MENU
:: 2. Environment variable override
if not "%DRIVEGUARD_PHONE_URL%"=="" (
    set "PHONE_URL=%DRIVEGUARD_PHONE_URL%"
    goto :RUN_STREAM
)

:: 3. Read last-used URL from launchers\phone_url.txt if available
set "LAST_URL="
if exist "%~dp0phone_url.txt" (
    set /p LAST_URL=<"%~dp0phone_url.txt"
)

:: Determine default choice number
set "DEFAULT_CHOICE=1"
if "!LAST_URL!"=="%URL_2%" (
    set "DEFAULT_CHOICE=2"
) else if not "!LAST_URL!"=="" (
    if not "!LAST_URL!"=="%URL_1%" (
        set "DEFAULT_CHOICE=3"
    )
)

echo ============================================================
echo   DriveGuard - Phone Camera Stream Launcher (Phase 2)
echo ============================================================
echo Select camera stream URL:
if "!DEFAULT_CHOICE!"=="1" (
    echo   [1] %URL_1% [Default]
) else (
    echo   [1] %URL_1%
)
if "!DEFAULT_CHOICE!"=="2" (
    echo   [2] %URL_2% [Default]
) else (
    echo   [2] %URL_2%
)
if "!DEFAULT_CHOICE!"=="3" (
    echo   [3] Custom URL [Previous: !LAST_URL!] [Default]
) else (
    echo   [3] Custom URL [enter manually]
)
echo ============================================================

set "CHOICE=!DEFAULT_CHOICE!"
set /p CHOICE="Enter choice [1, 2, or 3] (Default: !DEFAULT_CHOICE!): "

:: Extract first character in case of trailing spaces
set "FIRST_CHAR=!CHOICE:~0,1!"

if "!FIRST_CHAR!"=="1" (
    set "PHONE_URL=%URL_1%"
) else if "!FIRST_CHAR!"=="2" (
    set "PHONE_URL=%URL_2%"
) else if "!FIRST_CHAR!"=="3" (
    echo.
    set "CUSTOM_PROMPT=Enter custom stream URL: "
    if not "!LAST_URL!"=="" (
        set "CUSTOM_PROMPT=Enter custom stream URL [!LAST_URL!]: "
    )
    set "CUSTOM_URL="
    set /p CUSTOM_URL="!CUSTOM_PROMPT!"
    if not "!CUSTOM_URL!"=="" (
        set "PHONE_URL=!CUSTOM_URL!"
    ) else if not "!LAST_URL!"=="" (
        set "PHONE_URL=!LAST_URL!"
    ) else (
        set "PHONE_URL=%URL_1%"
    )
) else (
    :: User entered or pasted an explicit URL directly
    set "PHONE_URL=!CHOICE!"
)

:: Persist the selected URL
(> "%~dp0phone_url.txt" echo !PHONE_URL!)

:RUN_STREAM
echo.
echo Connecting to stream: %PHONE_URL%
echo Tip: Press 'm' in preview window to toggle mirror on/off
echo      Press 'q' or ESC to exit
echo ============================================================
echo.

".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url %PHONE_URL% %EXTRA_ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Camera runner exited with error code %ERRORLEVEL%.
    pause
)

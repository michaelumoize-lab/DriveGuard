@echo off
title DriveGuard - Next.js Frontend

:: Automatically resolve Project Root whether run from root or from launchers/
set "ROOT=%~dp0"
if not exist "%ROOT%frontend\package.json" (
    set "ROOT=%~dp0..\"
)
cd /d "%ROOT%frontend"

echo ============================================================
echo   DriveGuard - Starting Next.js Web Frontend
echo   Folder: %CD%
echo   URL:    http://localhost:3000
echo ============================================================
echo.

npm run dev

pause

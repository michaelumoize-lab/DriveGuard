# DriveGuard - Phone Camera Stream Launcher
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DriveGuard - Launching Phone Camera Stream (Phase 2)" -ForegroundColor Cyan
Write-Host "  Target: http://192.168.0.103:8080/video" -ForegroundColor Yellow
Write-Host "  Tip: Press 'm' in preview window to toggle mirror on/off" -ForegroundColor DarkGray
Write-Host "       Press 'q' or ESC to exit" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

& ".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url http://192.168.0.103:8080/video @args

# DriveGuard - Phone Camera Stream Launcher
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$DefaultUrl = "http://192.168.0.103:8080/video"
$ConfigFile = Join-Path $PSScriptRoot "phone_url.txt"
$PhoneUrl = ""

# 1. CLI argument override
if ($args.Count -gt 0 -and $args[0] -match "^(http|https|rtsp)://") {
    $PhoneUrl = $args[0]
}
# 2. Environment variable override
elseif ($env:DRIVEGUARD_PHONE_URL) {
    $PhoneUrl = $env:DRIVEGUARD_PHONE_URL
}
# 3. Read from configuration file
elseif (Test-Path $ConfigFile) {
    $PhoneUrl = (Get-Content $ConfigFile -TotalCount 1).Trim()
}

if (-not $PhoneUrl) {
    $PhoneUrl = $DefaultUrl
}

# If running interactively without a CLI argument override, allow quick confirmation or modification
if ($args.Count -eq 0) {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  DriveGuard - Phone Camera Stream Launcher (Phase 2)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Configured stream URL: $PhoneUrl" -ForegroundColor Yellow
    Write-Host ""
    $UserInput = Read-Host "Press ENTER to use configured URL, or type new URL"
    if ($UserInput -and $UserInput.Trim() -ne "") {
        $PhoneUrl = $UserInput.Trim()
        Set-Content -Path $ConfigFile -Value $PhoneUrl
    }
}

Write-Host "Connecting to phone stream: $PhoneUrl" -ForegroundColor Yellow
Write-Host "Tip: Press 'm' in preview window to toggle mirror on/off" -ForegroundColor DarkGray
Write-Host "     Press 'q' or ESC to exit" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

& ".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url $PhoneUrl @args

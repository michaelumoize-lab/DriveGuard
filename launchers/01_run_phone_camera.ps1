# DriveGuard - Phone Camera Stream Launcher
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Url1 = "http://192.168.0.135:4747/video"
$Url2 = "http://192.168.0.103:8080/video"
$ConfigFile = Join-Path $PSScriptRoot "phone_url.txt"
$PhoneUrl = ""
$ExtraArgs = @()

# 1. CLI argument override
if ($args.Count -gt 0) {
    $FirstArg = $args[0]
    if ($FirstArg -eq "1") {
        $PhoneUrl = $Url1
        if ($args.Count -gt 1) { $ExtraArgs = $args[1..($args.Count - 1)] }
    }
    elseif ($FirstArg -eq "2") {
        $PhoneUrl = $Url2
        if ($args.Count -gt 1) { $ExtraArgs = $args[1..($args.Count - 1)] }
    }
    elseif ($FirstArg -match "^(http|https|rtsp)://") {
        $PhoneUrl = $FirstArg
        if ($args.Count -gt 1) { $ExtraArgs = $args[1..($args.Count - 1)] }
    }
    else {
        $ExtraArgs = $args
    }
}

# 2. Environment variable override
if (-not $PhoneUrl -and $env:DRIVEGUARD_PHONE_URL) {
    $PhoneUrl = $env:DRIVEGUARD_PHONE_URL
}

# 3. Interactive prompt if no CLI or env override
if (-not $PhoneUrl) {
    $LastUrl = ""
    if (Test-Path $ConfigFile) {
        $LastUrl = (Get-Content $ConfigFile -TotalCount 1).Trim()
    }

    $DefaultChoice = "1"
    if ($LastUrl -eq $Url2) {
        $DefaultChoice = "2"
    }
    elseif ($LastUrl -and $LastUrl -ne $Url1) {
        $DefaultChoice = "3"
    }

    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  DriveGuard - Phone Camera Stream Launcher (Phase 2)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Select camera stream URL:" -ForegroundColor White
    
    if ($DefaultChoice -eq "1") {
        Write-Host "  [1] $Url1 [Default]" -ForegroundColor Green
    } else {
        Write-Host "  [1] $Url1" -ForegroundColor Green
    }

    if ($DefaultChoice -eq "2") {
        Write-Host "  [2] $Url2 [Default]" -ForegroundColor Green
    } else {
        Write-Host "  [2] $Url2" -ForegroundColor Green
    }

    if ($DefaultChoice -eq "3") {
        Write-Host "  [3] Custom URL [Previous: $LastUrl] [Default]" -ForegroundColor Yellow
    } else {
        Write-Host "  [3] Custom URL [enter manually]" -ForegroundColor Yellow
    }
    Write-Host "============================================================" -ForegroundColor Cyan

    $Choice = Read-Host "Enter choice [1, 2, or 3] (Default: $DefaultChoice)"
    if ([string]::IsNullOrWhiteSpace($Choice)) {
        $Choice = $DefaultChoice
    } else {
        $Choice = $Choice.Trim()
    }

    if ($Choice -eq "1") {
        $PhoneUrl = $Url1
    }
    elseif ($Choice -eq "2") {
        $PhoneUrl = $Url2
    }
    elseif ($Choice -eq "3") {
        Write-Host ""
        $CustomPrompt = if ($LastUrl) { "Enter custom stream URL [$LastUrl]" } else { "Enter custom stream URL" }
        $CustomInput = Read-Host $CustomPrompt
        if (-not [string]::IsNullOrWhiteSpace($CustomInput)) {
            $PhoneUrl = $CustomInput.Trim()
        }
        elseif ($LastUrl) {
            $PhoneUrl = $LastUrl
        }
        else {
            $PhoneUrl = $Url1
        }
    }
    else {
        # If user directly typed/pasted a URL
        $PhoneUrl = $Choice
    }

    # Persist the selected URL
    Set-Content -Path $ConfigFile -Value $PhoneUrl
}

Write-Host ""
Write-Host "Connecting to phone stream: $PhoneUrl" -ForegroundColor Yellow
Write-Host "Tip: Press 'm' in preview window to toggle mirror on/off" -ForegroundColor DarkGray
Write-Host "     Press 'q' or ESC to exit" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

& ".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url $PhoneUrl @ExtraArgs

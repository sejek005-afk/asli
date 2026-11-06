# ASLI one-shot installer for a Windows VPS.
# Run in PowerShell as Administrator:
#   Set-ExecutionPolicy Bypass -Scope Process -Force; iwr https://raw.githubusercontent.com/sejek005-afk/asli/main/windows/install.ps1 -UseBasicParsing | iex
#
# What it does: installs Python 3.12 and Git if missing, clones/updates the repo into C:\asli,
# creates a virtualenv, installs requirements, creates config.yaml from the example with a random
# API token, opens the firewall port, and registers two scheduled tasks:
#   ASLI-Live      — starts the bot at logon and restarts it if it stops
#   ASLI-Backtest  — re-runs the backtest + retraining every Sunday 03:00 (server time)

$ErrorActionPreference = "Stop"
$Root = "C:\asli"
$Repo = "https://github.com/sejek005-afk/asli.git"
$Port = 8787

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

Write-Host "== ASLI installer ==" -ForegroundColor Cyan

if (-not (Have "python")) {
    Write-Host "Installing Python 3.12..."
    if (Have "winget") {
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
    } else {
        $py = "$env:TEMP\python-3.12.7-amd64.exe"
        Invoke-WebRequest "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $py
        Start-Process $py -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
    }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

if (-not (Have "git")) {
    Write-Host "Installing Git..."
    if (Have "winget") {
        winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements --silent
    } else {
        $g = "$env:TEMP\git-setup.exe"
        Invoke-WebRequest "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe" -OutFile $g
        Start-Process $g -ArgumentList "/VERYSILENT /NORESTART" -Wait
    }
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
}

if (Test-Path "$Root\.git") {
    Write-Host "Updating $Root..."
    git -C $Root pull --ff-only
} else {
    git clone $Repo $Root
}

Set-Location $Root
if (-not (Test-Path "$Root\.venv")) { python -m venv .venv }
& "$Root\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$Root\.venv\Scripts\python.exe" -m pip install -r requirements.txt

if (-not (Test-Path "$Root\config.yaml")) {
    $bytes = New-Object byte[] 24
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    $token = ([Convert]::ToBase64String($bytes)).Replace("+", "A").Replace("/", "B").Replace("=", "")
    (Get-Content "$Root\config.example.yaml" -Raw).Replace("PUT-A-LONG-RANDOM-STRING-HERE", $token) |
        Set-Content "$Root\config.yaml" -Encoding UTF8
    Write-Host ""
    Write-Host "API token for the Android app: $token" -ForegroundColor Yellow
    Write-Host "(also saved in C:\asli\config.yaml under server.token)"
}

New-NetFirewallRule -DisplayName "ASLI dashboard" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -ErrorAction SilentlyContinue | Out-Null

$user = "$env:USERDOMAIN\$env:USERNAME"
$live = New-ScheduledTaskAction -Execute "$Root\windows\start_live.cmd" -WorkingDirectory $Root
$trigLogon = New-ScheduledTaskTrigger -AtLogOn -User $user
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 3650)
Register-ScheduledTask -TaskName "ASLI-Live" -Action $live -Trigger $trigLogon -Settings $set -User $user -RunLevel Highest -Force | Out-Null

$bt = New-ScheduledTaskAction -Execute "$Root\windows\weekly_backtest.cmd" -WorkingDirectory $Root
$trigWeek = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am
Register-ScheduledTask -TaskName "ASLI-Backtest" -Action $bt -Trigger $trigWeek -Settings $set -User $user -RunLevel Highest -Force | Out-Null

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host " 1) Open MetaTrader 5, log in to your DEMO account, and in Tools > Options > Expert Advisors tick 'Allow algorithmic trading'."
Write-Host "    In Tools > Options > Charts set 'Max bars in chart' to Unlimited."
Write-Host " 2) notepad C:\asli\config.yaml  -> set mt5.login, mt5.server, symbols (exact names from Market Watch)."
Write-Host "    Password: setx ASLI_MT5_PASSWORD ""your-demo-password""   (then open a NEW PowerShell)"
Write-Host " 3) C:\asli\windows\run_backtest.cmd     (first backtest + training, takes a few minutes)"
Write-Host " 4) Start-ScheduledTask ASLI-Live      (or log off/on). Dashboard: http://<server-ip>:$Port/"

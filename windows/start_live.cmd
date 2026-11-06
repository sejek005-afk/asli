@echo off
REM Starts the ASLI live bot and restarts it if it exits (MT5 disconnect, weekly retrain, crash).
cd /d C:\asli
if not exist logs mkdir logs
:loop
"C:\asli\.venv\Scripts\python.exe" run_live.py >> logs\console.log 2>&1
echo %date% %time% bot exited, restarting in 30s >> logs\console.log
timeout /t 30 /nobreak > nul
goto loop

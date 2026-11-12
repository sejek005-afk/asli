@echo off
REM Weekly: pull latest code, refresh history, re-run research and retrain, then restart the bot
REM (start_live.cmd's loop restarts python automatically after it is stopped).
cd /d C:\asli
if not exist logs mkdir logs
git pull --ff-only >> logs\weekly.log 2>&1
"C:\asli\.venv\Scripts\python.exe" -m pip install -r requirements.txt >> logs\weekly.log 2>&1
"C:\asli\.venv\Scripts\python.exe" run_backtest.py --years 6 --refresh >> logs\weekly.log 2>&1
if exist state\live.pid (
  for /f %%p in (state\live.pid) do taskkill /F /PID %%p >> logs\weekly.log 2>&1
)

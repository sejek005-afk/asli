@echo off
REM Backtest all 5 styles + fusion on MT5 history, retrain the learner, write reports\report.html
cd /d C:\asli
if not exist logs mkdir logs
"C:\asli\.venv\Scripts\python.exe" run_backtest.py --years 6 %*
start "" "C:\asli\reports\report.html"
pause

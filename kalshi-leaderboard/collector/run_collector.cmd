@echo off
rem Restart loop for the Kalshi collector. Launched by the KalshiLeaderboardCollector task.
cd /d "%~dp0.."
:loop
echo %date% %time% starting collector >> logs\task_stdout.log
python -I collector\collector.py >> logs\task_stdout.log 2>&1
echo %date% %time% collector exited %errorlevel%, restarting in 15s >> logs\task_stdout.log
timeout /t 15 /nobreak >nul
goto loop

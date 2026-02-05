@echo off
setlocal

REM Try 'training' service first
set SERVICE=training

docker compose ps -q %SERVICE% >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo Attaching to running '%SERVICE%' service...
    docker compose exec -it %SERVICE% python scripts/monitor_dashboard.py
) else (
    echo Service '%SERVICE%' is not running ^(it has 'donotautostart' profile^).
    echo Starting a temporary container for monitoring...
    docker compose run --rm %SERVICE% python scripts/monitor_dashboard.py
)

endlocal

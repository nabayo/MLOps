@echo off
REM Export MLflow data to a backup zip file using Docker (Windows batch version)

setlocal enabledelayedexpansion

REM Parse arguments
set OUTPUT_NAME=%1

echo [*] Exporting MLflow data via Docker...
echo.

REM Run export
if "%OUTPUT_NAME%"=="" (
    docker-compose run --rm export
) else (
    docker-compose run --rm -e BACKUP_NAME=%OUTPUT_NAME% export python scripts/export_mlflow.py --output-dir /app/backups --name %OUTPUT_NAME%
)

echo.
echo [+] Export complete! Backup saved in .\backups\
dir /b /o-d backups\*.zip | findstr /n "^" | findstr "^1:"

endlocal

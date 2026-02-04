@echo off
REM Import MLflow data from a backup zip file using Docker (Windows batch version)

setlocal enabledelayedexpansion

if "%1"=="" (
    echo [!] Error: Backup file not specified
    echo.
    echo Usage:
    echo   scripts\docker_import.bat ^<backup_file.zip^> [--dry-run] [--overwrite]
    echo.
    echo Examples:
    echo   scripts\docker_import.bat my_backup.zip
    echo   scripts\docker_import.bat my_backup.zip --dry-run
    echo   scripts\docker_import.bat my_backup.zip --overwrite
    echo.
    exit /b 1
)

set BACKUP_FILE=%1
shift

REM Build additional arguments
set EXTRA_ARGS=
:parse_args
if "%1"=="" goto args_done
set EXTRA_ARGS=%EXTRA_ARGS% %1
shift
goto parse_args
:args_done

REM Check if file exists
if not exist "backups\%BACKUP_FILE%" (
    echo [!] Error: Backup file not found: backups\%BACKUP_FILE%
    echo.
    echo Available backups:
    dir /b backups\*.zip 2>nul || echo   (none)
    exit /b 1
)

echo [*] Importing MLflow data from: %BACKUP_FILE%
echo.

REM Run import
set BACKUP_FILE=%BACKUP_FILE%
docker-compose run --rm -e BACKUP_FILE=%BACKUP_FILE% import python scripts/import_mlflow.py /app/backups/%BACKUP_FILE% %EXTRA_ARGS%

echo.
echo [+] Import complete!

endlocal

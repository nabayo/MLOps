@echo off
setlocal

echo 🐳 Running Model Registration inside Docker...

:: Check if docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running.
    exit /b 1
)

:: Run script using docker compose
docker compose run --rm --entrypoint python training scripts/register_models.py

if %errorlevel% equ 0 (
    echo ✅ Registration complete!
) else (
    echo ❌ Registration failed.
    exit /b 1
)

endlocal

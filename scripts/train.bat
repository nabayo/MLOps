@echo off
REM Quick training script using Docker (Windows batch version)

echo ==============================================
echo   MLOps Training Pipeline with Docker
echo ==============================================
echo.

REM Check if services are running
docker ps | findstr mlops-mlflow >nul 2>&1
if errorlevel 1 (
    echo [!] MLflow not running. Starting services...
    docker-compose up -d
    echo [*] Waiting for services to be healthy...
    timeout /t 10 /nobreak >nul
)

REM Build training image if needed
echo [*] Building training image...
docker-compose build training

REM Run training
echo [*] Running training...
docker-compose run --rm training python main.py train --evaluate

echo.
echo ===================================
echo   Training complete!
echo ===================================
echo View results at: http://localhost:5000

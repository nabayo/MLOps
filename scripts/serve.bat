@echo off
REM Start the MLOps serving API (Windows batch version)

echo =================================
echo   MLOps Serving API
echo =================================
echo.

REM Check if MLflow is running
docker ps | findstr mlops-mlflow >nul 2>&1
if errorlevel 1 (
    echo [!] MLflow not running. Starting infrastructure services...
    docker-compose up -d mlflow postgres minio
    echo [*] Waiting for services to be healthy...
    timeout /t 5 /nobreak >nul
)

REM Start serving API
echo [*] Building serving image...
docker-compose build serving frontend

echo [*] Starting serving API...
docker-compose up -d serving frontend

echo.
echo ================================
echo   Serving API is running!
echo ================================
echo API endpoint: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo MLflow UI: http://localhost:5000
echo.
echo Test the API:
echo   curl http://localhost:8000/health
echo.
echo To stop: docker-compose stop serving frontend

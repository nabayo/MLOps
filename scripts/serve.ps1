#!/usr/bin/env pwsh
# Start the MLOps serving API (PowerShell version)

Write-Host "=================================" -ForegroundColor Cyan
Write-Host "  MLOps Serving API"
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if MLflow is running
$mlflowRunning = docker ps | Select-String "mlops-mlflow"
if (-not $mlflowRunning) {
    Write-Host "[!] MLflow not running. Starting infrastructure services..." -ForegroundColor Yellow
    docker-compose up -d mlflow postgres minio
    Write-Host "[*] Waiting for services to be healthy..."
    Start-Sleep -Seconds 5
}

# Start serving API
Write-Host "[*] Building serving image..." -ForegroundColor Cyan
docker-compose build serving

Write-Host "[*] Starting serving API..." -ForegroundColor Green
docker-compose up -d serving

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "  Serving API is running!"
Write-Host "================================" -ForegroundColor Green
Write-Host "[POINT] API endpoint: http://localhost:8000"
Write-Host "[DOCS]  API docs:     http://localhost:8000/docs"
Write-Host "[UI]    MLflow UI:    http://localhost:5000"
Write-Host ""
Write-Host "[INFO] Test the API:" -ForegroundColor Yellow
Write-Host "   curl http://localhost:8000/health"
Write-Host ""
Write-Host "[STOP] To stop: docker-compose stop serving" -ForegroundColor Gray

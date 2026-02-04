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

# Check if serving image exists
$imageExists = docker images | Select-String "mlops-serving"

if (-not $imageExists) {
    Write-Host "[*] Building serving image (first time)..." -ForegroundColor Cyan
    docker-compose build serving frontend
} else {
    Write-Host "[OK] Serving image exists, skipping build" -ForegroundColor Green
    Write-Host "     To rebuild: docker-compose build serving" -ForegroundColor Gray
}

Write-Host "[*] Starting serving API..." -ForegroundColor Green
docker-compose up -d serving frontend

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "  Serving API is running!"
Write-Host "================================" -ForegroundColor Green
Write-Host "API endpoint: http://localhost:8000"
Write-Host "API docs:     http://localhost:8000/docs"
Write-Host "MLflow UI:    http://localhost:5000"
Write-Host ""
Write-Host "Test the API:" -ForegroundColor Yellow
Write-Host "   curl http://localhost:8000/health"
Write-Host ""
Write-Host "To stop: docker-compose stop serving frontend" -ForegroundColor Gray

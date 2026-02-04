#!/usr/bin/env pwsh
# Quick training script using Docker (PowerShell version)

Write-Host "=============================================="
Write-Host "  MLOps Training Pipeline with Docker"
Write-Host "=============================================="
Write-Host ""

# Check if services are running
$mlflowRunning = docker ps | Select-String "mlops-mlflow"
if (-not $mlflowRunning) {
    Write-Host "[!] MLflow not running. Starting services..." -ForegroundColor Yellow
    docker-compose up -d
    Write-Host "[*] Waiting for services to be healthy..."
    Start-Sleep -Seconds 10
}

# Build training image if needed
Write-Host "[*] Building training image..." -ForegroundColor Cyan
docker-compose build training

# Run training
Write-Host "[*] Running training..." -ForegroundColor Green
docker-compose run --rm training python main.py train --evaluate

Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "  Training complete!"
Write-Host "===================================" -ForegroundColor Green
Write-Host "View results at: http://localhost:5000"

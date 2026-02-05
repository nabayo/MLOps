Write-Host "🔍 Checking Model Registry State..." -ForegroundColor Cyan

# Check if docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running."
    exit 1
}

# Run script using docker compose
docker compose run --rm --entrypoint python training scripts/check_registry.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Check complete!" -ForegroundColor Green
} else {
    Write-Host "❌ Check failed." -ForegroundColor Red
    exit 1
}

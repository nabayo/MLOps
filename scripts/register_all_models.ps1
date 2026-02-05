Write-Host "🐳 Running Model Registration inside Docker..." -ForegroundColor Cyan

# Check if docker is running
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker is not running."
    exit 1
}

# Run script using docker compose
# We use docker compose run to ensure network and env vars are present
docker compose run --rm --entrypoint python training scripts/register_models.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Registration complete!" -ForegroundColor Green
} else {
    Write-Host "❌ Registration failed." -ForegroundColor Red
    exit 1
}

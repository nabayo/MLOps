# Launch the Monitoring Dashboard inside Docker

$ErrorActionPreference = "SilentlyContinue"
$Service = "training"

# Check if 'training' container is running
if (docker compose ps -q $Service) {
    Write-Host "Attaching to running '$Service' service..." -ForegroundColor Green
    docker compose exec -it $Service python scripts/monitor_dashboard.py
} else {
    Write-Host "Service '$Service' is not running (it has 'donotautostart' profile)." -ForegroundColor Yellow
    Write-Host "Starting a temporary container for monitoring..." -ForegroundColor WhatsApp
    docker compose run --rm $Service python scripts/monitor_dashboard.py
}

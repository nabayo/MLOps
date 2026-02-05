#!/bin/bash
# Launch the Monitoring Dashboard inside Docker

# Try to use 'training' service first, as it has all dependencies mounted
SERVICE="training"

# Check if 'training' container is running
if [ -n "$(docker compose ps -q $SERVICE)" ]; then
    echo "Attaching to running '$SERVICE' service..."
    docker compose exec -it $SERVICE python scripts/monitor_dashboard.py
else
    echo "Service '$SERVICE' is not running (it has 'donotautostart' profile)."
    echo "Starting a temporary container for monitoring..."
    docker compose run --rm $SERVICE python scripts/monitor_dashboard.py
fi

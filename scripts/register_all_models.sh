#!/bin/bash

# Wrapper to run the register_models.py script inside the MLflow Docker environment

echo "🐳 Running Model Registration inside Docker..."

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running."
  exit 1
fi

# Use the training container as it has access to volumes and environment
# We use 'run --rm' to spin up a temporary container or 'exec' if it's already running?
# 'training' service profile is 'donotautostart', so it might not be running.
# However, 'mlops-training' container name is defined.

# Safe approach: usage `docker compose run --rm training ...`
# This ensures environment variables and networks are set up correctly without relying on an existing container.

docker compose run --rm --entrypoint python training scripts/register_models.py

if [ $? -eq 0 ]; then
    echo "✅ Registration complete!"
else
    echo "❌ Registration failed."
    exit 1
fi

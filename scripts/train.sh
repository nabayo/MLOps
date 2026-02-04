#!/bin/bash
# Quick training script using Docker

echo "🚀 Starting MLOps Training Pipeline with Docker"
echo "================================================"

# Check if services are running
if ! docker ps | grep -q mlops-mlflow; then
    echo "⚠️  MLflow not running. Starting services..."
    docker-compose up -d
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
fi

# Build training image if needed
echo "🔨 Building training image..."
docker-compose build training

# Run training
echo "🏋️  Running training..."
docker-compose run --rm training python main.py train --evaluate

echo ""
echo "✅ Training complete!"
echo "📊 View results at: http://localhost:5000"

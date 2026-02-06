#!/bin/bash
# Training script using Docker with CUDA support

echo "🚀 Starting MLOps Training Pipeline with Docker (CUDA Enabled)"
echo "============================================================"

# Check if services are running
if ! docker ps | grep -q mlops-mlflow; then
    echo "⚠️  MLflow not running. Starting services..."
    docker-compose up -d
    echo "⏳ Waiting for services to be healthy..."
    sleep 10
fi

# Build training image if needed
echo "🔨 Building CUDA training image..."
docker-compose -f compose.yml -f compose.cuda.yml build training

# Run training
echo "🏋️  Running training on GPU..."
docker-compose -f compose.yml -f compose.cuda.yml run --rm training python main.py train --evaluate

echo ""
echo "✅ Training complete!"
echo "📊 View results at: http://localhost:5000"

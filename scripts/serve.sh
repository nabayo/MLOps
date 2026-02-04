#!/bin/bash
# Start the MLOps serving API

set -e

echo "🚀 Starting MLOps Serving API"
echo "================================"

# Check if MLflow is running
if ! docker ps | grep -q mlops-mlflow; then
    echo "⚠️  MLflow not running. Starting infrastructure services..."
    docker-compose up -d mlflow postgres minio
    echo "⏳ Waiting for services to be healthy..."
    sleep 5
fi

# Start serving API
echo "🔨 Building serving image..."
docker-compose build serving

echo "🌐 Starting serving API..."
docker-compose up -d serving

echo ""
echo "✅ Serving API is running!"
echo "================================"
echo "📍 API endpoint: http://localhost:8000"
echo "📖 API docs: http://localhost:8000/docs"
echo "📊 MLflow UI: http://localhost:5000"
echo ""
echo "💡 Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "🛑 To stop: docker-compose stop serving"

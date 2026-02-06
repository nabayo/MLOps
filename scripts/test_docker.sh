#!/bin/bash
set -e

# Configuration
IMAGE_NAME="mlops-serving"
CONTAINER_NAME="mlops-test-runner"
NETWORK="mlops_mlops-network"
DOCKERFILE="serving/Dockerfile"

# Build the image
echo "🏗 Building Docker image..."
docker build -t $IMAGE_NAME -f $DOCKERFILE .

# Create output directory
mkdir -p test_results

# Run the container
echo "🚀 Running test in Docker..."
echo "   Network: $NETWORK"
echo "   Volumes: dataset, models, test_results"

docker run --rm \
    --name $CONTAINER_NAME \
    --network $NETWORK \
    -v "$(pwd)/dataset:/app/dataset" \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/experiments:/app/experiments" \
    -v "$(pwd)/test_results:/app/test_results" \
    -e MLFLOW_TRACKING_URI="http://mlflow:5000" \
    -e AWS_ACCESS_KEY_ID="minioadmin" \
    -e AWS_SECRET_ACCESS_KEY="minioadmin_password_change_me" \
    -e MLFLOW_S3_ENDPOINT_URL="http://minio:9000" \
    $IMAGE_NAME \
    python serving/test.py "$@"

echo "✅ Test completed successfully!"

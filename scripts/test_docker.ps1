$ErrorActionPreference = "Stop"

# Configuration
$IMAGE_NAME = "mlops-serving"
$CONTAINER_NAME = "mlops-test-runner"
$NETWORK = "mlops_mlops-network"
$DOCKERFILE = "serving/Dockerfile"

# Build the image
Write-Host "🏗 Building Docker image..." -ForegroundColor Cyan
docker build -t $IMAGE_NAME -f $DOCKERFILE .

# Create output directory
New-Item -ItemType Directory -Force -Path "test_results" | Out-Null

# Run the container
Write-Host "🚀 Running test in Docker..." -ForegroundColor Cyan
Write-Host "   Network: $NETWORK"
Write-Host "   Volumes: dataset, models, test_results"

docker run --rm `
    --name $CONTAINER_NAME `
    --network $NETWORK `
    -v "${PWD}/dataset:/app/dataset" `
    -v "${PWD}/models:/app/models" `
    -v "${PWD}/experiments:/app/experiments" `
    -v "${PWD}/test_results:/app/test_results" `
    -e MLFLOW_TRACKING_URI="http://mlflow:5000" `
    -e AWS_ACCESS_KEY_ID="minioadmin" `
    -e AWS_SECRET_ACCESS_KEY="minioadmin_password_change_me" `
    -e MLFLOW_S3_ENDPOINT_URL="http://minio:9000" `
    $IMAGE_NAME `
    python serving/test.py $args

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Test completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Test failed!" -ForegroundColor Red
}

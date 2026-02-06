@echo off
setlocal

:: Configuration
set IMAGE_NAME=mlops-serving
set CONTAINER_NAME=mlops-test-runner
set NETWORK=mlops-network
set DOCKERFILE=serving/Dockerfile

:: Build the image
echo 🏗 Building Docker image...
docker build -t %IMAGE_NAME% -f %DOCKERFILE% .
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%

:: Create output directory
if not exist "test_results" mkdir "test_results"

:: Run the container
echo 🚀 Running test in Docker...
echo    Network: %NETWORK%
echo    Volumes: dataset, models, test_results

docker run --rm ^
    --name %CONTAINER_NAME% ^
    --network %NETWORK% ^
    -v "%CD%/dataset:/app/dataset" ^
    -v "%CD%/models:/app/models" ^
    -v "%CD%/test_results:/app/test_results" ^
    -e MLFLOW_TRACKING_URI="http://mlflow:5000" ^
    -e AWS_ACCESS_KEY_ID="minio" ^
    -e AWS_SECRET_ACCESS_KEY="minio123" ^
    -e MLFLOW_S3_ENDPOINT_URL="http://minio:9000" ^
    %IMAGE_NAME% ^
    python serving/test.py %*

if %ERRORLEVEL% EQU 0 (
    echo ✅ Test completed successfully!
) else (
    echo ❌ Test failed!
)

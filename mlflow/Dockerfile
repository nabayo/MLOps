FROM python:3.12-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install \
    setuptools \
    mlflow==2.18.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.35.0

# Expose MLflow port
EXPOSE 5000

# Start MLflow server
CMD mlflow server \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port 5000

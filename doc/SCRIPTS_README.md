# Training Scripts

This directory contains helper scripts for running the MLOps platform.

## 📋 Available Scripts

### 🏋️ Training
- **`train.sh`** / **`train.ps1`** / **`train.bat`** - Run model training with Docker

### 🌐 Serving
- **`serve.sh`** / **`serve.ps1`** / **`serve.bat`** - Start the inference API server

### 📤 Export (Backup)
- **`docker_export.sh`** / **`docker_export.ps1`** / **`docker_export.bat`** - Export MLflow data to zip
- **`export_mlflow.py`** - Python script for exporting (used by Docker)

### 📥 Import (Restore)
- **`docker_import.sh`** / **`docker_import.ps1`** / **`docker_import.bat`** - Import MLflow data from zip
- **`import_mlflow.py`** - Python script for importing (used by Docker)

## 🖥️ Platform Support

| Script Type | Linux/Mac | Windows (CMD) | Windows (PowerShell) |
|-------------|-----------|---------------|----------------------|
| Training | `train.sh` | `train.bat` | `train.ps1` |
| Serving | `serve.sh` | `serve.bat` | `serve.ps1` |
| Export | `docker_export.sh` | `docker_export.bat` | `docker_export.ps1` |
| Import | `docker_import.sh` | `docker_import.bat` | `docker_import.ps1` |

## 🚀 Quick Start

### Linux / Mac

```bash
# Training
./scripts/train.sh
- Start MLflow services if not running
- Build training container
- Run training with evaluation
- Log everything to MLflow

### Custom Training Commands

```bash
# Basic training
docker-compose run --rm training python main.py train

# Training with evaluation
docker-compose run --rm training python main.py train --evaluate

# Evaluation only
docker-compose run --rm training python main.py eval --model /app/models/best.pt

# Custom config
docker-compose run --rm training python main.py train --training-config /app/configs/custom.yaml
```

### Quick Commands

```bash
# Just build the training image
docker-compose build training

# Run training in background
docker-compose up -d training

# View training logs
docker-compose logs -f training
```

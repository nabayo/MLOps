# YOLOv11 Finger Counting MLOps Platform

Public github repository: https://github.com/nabayo/MLOps.

This project implements an end-to-end MLOps pipeline for training, tracking, and serving a YOLOv11-based finger counting model. It leverages MLflow for experiment management, Docker for containerization, and provides a real-time web interface for inference.

## Architecture

The system is composed of the following services:

1.  **MLflow Tracking Server:** Centralized repository for experiment metadata, metrics, and parameters.
2.  **PostgreSQL Database:** Backend storage for MLflow metadata.
3.  **MinIO Object Storage:** S3-compatible storage for model artifacts (weights, plots).
4.  **Training Pipeline:** Modular Python scripts for data preparation (from Picsellia), YOLOv11 training, and model evaluation.
5.  **Serving API:** FastAPI application that dynamically loads models from the MLflow registry for inference.
6.  **Frontend Web App:** Frontend interface for real-time webcam inference and experiment visualization, communicating with the backend via WebSocket and REST APIs.

## Prerequisites

-   Docker and Docker Compose
-   Python 3.12+ (for local development)
-   Picsellia API Token (required for dataset download)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:nabayo/MLOps.git
    cd MLOps
    ```

2.  **Configure Environment:**
    Copy the example environment file and update variables as needed. (You can use the default ones without any issues).
    ```bash
    cp .env.example .env
    ```
    **If you want to start training:** Ensure there is a file in the root directory named `picsellia_token` containing your valid API token.


3. **Import backup**:

    To import all the models weights and artifacts from a training session, there is a backup import/export system where you can export your training session into a backup zip file (e.g., `mlflow_backup_20260206_152906.zip`) that you can import back into another system:

    1.  **Place the backup file** into the `backups/` directory in the project root. Create that folder if it doesn't exist.
    2.  Run the import script with the filename.

    **Linux/MacOS**
    ```bash
    ./scripts/import_backup.sh mlflow_backup_20260206_152906.zip
    ```
    *(Note: You can provide just the filename or the path `backups/filename.zip`, but the file MUST be inside the `backups/` folder).*

    **Windows**
    ```powershell
    ./scripts/import_backup.ps1 mlflow_backup_20260206_152906.zip
    ```

4. **Start the web app**:

    Launch the full stack using Docker Compose.
    ```bash
    docker-compose up -d
    ```

    Or launch it via the scripts:

    **Linux/MacOS**
    ```bash
    ./scripts/serve.sh
    ```

    **Windows**
    ```powershell
    ./scripts/serve.ps1
    ```

## Usage

### Web Dashboard
Access the dashboard at `http://localhost:8080`.

-   **Live Inference:** Use the webcam to perform real-time finger counting. Adjust confidence thresholds and frame skipping parameters.
-   **Experiments / Model Selection:** Browse and load different model versions from the MLflow registry without restarting the server. View training history and metrics for all experiments.
-   **Custom Image Upload:** Upload your own image to perform inference on that image.

### Monitoring Experiments

With the docker up and running, you can access to:

-   **MLflow UI:** `http://localhost:5000`
-   **MinIO:** `http://localhost:9001`

### Training a Model

There are two config files for training:

- `config.yaml`: Contain the dataset name, version and local target download path.
- `training_config.yaml`: Contain the training parameters, like the number of epochs, batch size, model version, image augmentation parameters, etc.

To start a training job using the Docker container:

```bash
docker-compose up -d training
```

Or launch it via the scripts:

**Linux/MacOS**
```bash
./scripts/train.sh
```

**Windows**
```bash
./scripts/train.ps1
```

This process will:
1.  Download the dataset from Picsellia.
2.  Preprocess and split the data.
3.  Train the YOLO model.
4.  Log metrics and register the trained model in MLflow.

### API Endpoints
The backend API is available at `http://localhost:8000`.

-   `GET /health`: Check service status.
-   `GET /models/list`: List available models.
-   `POST /models/load`: Load a specific model version.
-   `POST /predict`: Perform inference on an uploaded image.
-   `websocket /ws/predict`: Real-time inference stream.

Interactive documentation is available at `http://localhost:8000/docs`.

## Backup Export after training

To save your current experiments and models:

**Linux/MacOS**
```bash
# Auto-generated name
./scripts/export_backup.sh

# Custom name
./scripts/export_backup.sh --name my_custom_backup
```

**Windows**
```powershell
# Auto-generated name
./scripts/export_backup.ps1

# Custom name
./scripts/export_backup.ps1 my_custom_backup
```

The backup zip file will be created in the `backups/` directory.

## Project Structure

-   `src/`: Core logic for data preparation, training, and evaluation.
-   `serving/`: FastAPI application code.
-   `frontend/`: Web application source code.
-   `configs/`: Configuration files for training hyperparameters and dataset settings.
-   `mlflow/`: Docker configuration for the MLflow server.

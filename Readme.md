# YOLOv11 Finger Counting - MLOps Platform

Production-grade MLOps platform for training, tracking, and serving YOLOv11 finger detection models with comprehensive experiment management.

## 🚀 Features

- **Complete MLOps Infrastructure**: MLflow for experiment tracking, model registry, and metadata storage
- **Configurable Training Pipeline**: Modular YOLOv11 training with extensive hyperparameter configuration
- **Dynamic Model Serving**: FastAPI backend with hot-swappable models from MLflow registry
- **Real-time Web Dashboard**: 
  - Live webcam inference with bounding box overlay
  - Interactive model selection and switching
  - Comprehensive experiments browser with metrics
- **Modular Preprocessing**: Face blur using deface library (configurable)
- **Self-hosted & Free**: 100% open-source, runs entirely in Docker

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Training a Model](#training-a-model)
- [Using the Web Dashboard](#using-the-web-dashboard)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MLOps Platform                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Training   │───▶│   MLflow     │◀──────│   Serving   │ │
│  │   Pipeline   │    │   Tracking   │    │     API     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                    │                    │        │
│         │                    │                    │        │
│         ▼                    ▼                    ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Picsellia   │    │  PostgreSQL  │    │    Web       │ │
│  │   Dataset    │    │    MinIO     │    │  Dashboard   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **MLflow Server**: Experiment tracking, metrics logging, model registry
2. **PostgreSQL**: MLflow backend store (experiments metadata)
3. **MinIO**: S3-compatible artifact store (model weights, plots, logs)
4. **Training Pipeline**: YOLOv11 training with comprehensive MLflow integration
5. **Serving API**: FastAPI backend with dynamic model loading
6. **Web Dashboard**: Real-time inference UI with experiment browser

## 🎯 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.12+ (for local development)
- Picsellia API token (store in `picsellia_token` file)

### 1. Clone and Setup

```bash
git clone <your-repo>
cd MLOps

# Copy environment template
cp .env.example .env

# Edit .env if needed (default values work for local development)
nano .env
```

### 2. Start MLflow Infrastructure

```bash
# Start MLflow, PostgreSQL, and MinIO
docker-compose up -d mlflow postgres minio

# Wait for services to be healthy (~30 seconds)
docker-compose ps

# Access MLflow UI
open http://localhost:5000
```

### 3. Train Your First Model

```bash
# Install dependencies (local development)
pip install -r requirements.txt

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Run training pipeline
python main.py train --evaluate

# Or use Docker
docker-compose run --rm training
```

Training will:
- ✅ Download dataset from Picsellia
- ✅ Prepare data (convert to YOLO format, create splits 80/10/10)
- ✅ Train YOLOv11 with configured hyperparameters
- ✅ Log all metrics, parameters, and artifacts to MLflow
- ✅ Register best model in MLflow Model Registry
- ✅ Evaluate on test set

### 4. Start Serving & Frontend

```bash
# Start all services
docker-compose up -d

# Access the web dashboard
open http://localhost:8080

# Access API docs
open http://localhost:8000/docs
```

## 🏋️ Training a Model

### Configuration

Edit `configs/training_config.yaml` to customize:

```yaml
model:
  architecture: "yolo11n"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  pretrained: true

dataset:
  split_ratios:
    train: 0.8  # Configurable!
    val: 0.1
    test: 0.1
  seed: 42
  img_size: 640

training:
  epochs: 100
  batch_size: 16
  lr0: 0.01
  # ... many more parameters

augmentation:
  fliplr: 0.5
  mosaic: 1.0
  # ... extensive augmentation options

mlflow:
  experiment_name: "yolov11-finger-counting"
  auto_register: true
  auto_promote_stage: "Staging"
```

### Run Training

**Local (Recommended for Development)**:
```bash
# Basic training
python main.py train

# Training with evaluation
python main.py train --evaluate

# Custom config
python main.py train --training-config configs/custom_config.yaml
```

**Docker (Production)**:
```bash
# Start training in background
docker-compose up -d training

# View logs
docker-compose logs -f training

# Stop after completion
docker-compose down training
```

### Monitor Training

1. **MLflow UI**: http://localhost:5000
   - View real-time metrics
   - Compare experiments
   - Download artifacts

2. **Logs**: Check console output for progress

### Evaluation Only

```bash
python main.py eval --model experiments/yolo11n_*/weights/best.pt
```

## 🌐 Using the Web Dashboard

Access: http://localhost:8080

### Page 1: Live Inference

1. Click **"Start Camera"** to enable webcam
2. Show your hand with different finger counts
3. View:
   - Real-time bounding boxes on video
   - Total finger count (sum of all hands)
   - FPS and latency stats
   - Individual predictions list

### Page 2: Model Selection

1. Browse all registered models from MLflow
2. View model versions with:
   - Architecture (yolo11n, yolo11s, etc.)
   - Stage (Production, Staging, Archived)
   - Metrics (mAP@50-95, precision, recall)
3. **Live Switch**: Click "Load Model" to switch without restart
4. Currently loaded model shows in Live Inference page

### Page 3: Experiments Dashboard

1. View all MLflow experiments
2. Browse runs with full metrics:
   - Model architecture
   - mAP@50-95, precision, recall
   - Training date and status
   - Hyperparameters
3. Compare different training runs
4. Click experiment in MLflow UI for detailed artifacts

## 📡 API Documentation

### Base URL

```
http://localhost:8000
```

### Endpoints

#### Health Check
```bash
GET /health
```

#### Get Current Model
```bash
GET /models/current
```

#### List All Models
```bash
GET /models/list
```

#### Load Model (Live Switching!)
```bash
POST /models/load?model_name=yolov11-finger-counting&version=1
POST /models/load?model_name=yolov11-finger-counting&stage=Production
```

#### List Experiments
```bash
GET /models/experiments
```

#### Predict (Inference)
```bash
POST /predict
Content-Type: multipart/form-data
Body: file=<image>

Response:
{
  "finger_count": 5,
  "predictions": [
    {
      "class_name": "5",
      "class_id": 4,
      "confidence": 0.95,
      "bbox": [100, 150, 200, 300]
    }
  ],
  "preprocessing_applied": false,
  "inference_time_ms": 45.2
}
```

### Interactive API Docs

Visit http://localhost:8000/docs for Swagger UI

## ⚙️ Configuration

### Dataset Configuration

`configs/config.yaml`:
```yaml
dataset_name: "Photos Floutées"
dataset_version: "new_version"
dataset_download_path: "dataset/"
```

### Training Configuration

See `configs/training_config.yaml` for:
- ✅ Model architecture (easily switch yolo11n/s/m/l/x)
- ✅ Dataset splits (configurable ratios)
- ✅ Training hyperparameters
- ✅ Data augmentation settings
- ✅ MLflow experiment settings

### Environment Variables

`.env`:
```bash
# PostgreSQL
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minio_password

# Serving
MODEL_NAME=yolov11-finger-counting
MODEL_STAGE=Production
ENABLE_PREPROCESSING=true  # Face blur
```

## 🛠️ Development

### Project Structure

```
MLOps/
├── configs/
│   ├── config.yaml              # Dataset configuration
│   └── training_config.yaml     # Training hyperparameters
├── src/
│   ├── config.py                # Config loader
│   ├── picsellia.py             # Dataset download
│   ├── data_preparation.py      # Data prep & YOLO conversion
│   ├── training.py              # Training with MLflow
│   ├── evaluation.py            # Model evaluation
│   └── preprocessing/
│       ├── base.py              # Preprocessing pipeline
│       └── face_blur.py         # Face anonymization
├── serving/
│   ├── api.py                   # FastAPI serving
│   └── Dockerfile
├── frontend/
│   ├── index.html               # Web UI
│   ├── app.js                   # JavaScript logic
│   ├── style.css                # Premium styling
│   ├── nginx.conf
│   └── Dockerfile
├── mlflow/
│   └── Dockerfile               # MLflow server
├── main.py                      # Pipeline orchestrator
├── docker-compose.yml           # All services
└── requirements.txt             # Dependencies
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start only MLflow infrastructure
docker-compose up -d mlflow postgres minio

# Run training locally
export MLFLOW_TRACKING_URI=http://localhost:5000
python main.py train

# Start serving locally
MLFLOW_TRACKING_URI=http://localhost:5000 python serving/api.py

# Serve frontend locally (simple HTTP server)
cd frontend && python -m http.server 8080
```

### Adding YOLO26 Support (Future)

When YOLO26 is released, simply update `src/training.py`:

```python
model_map = {
    # Existing
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    # ... other yolo11
    
    # Add YOLO26
    'yolo26n': 'yolo26n.pt',
    'yolo26s': 'yolo26s.pt',
    'yolo26m': 'yolo26m.pt',
    # ...
}
```

Then use in `training_config.yaml`:
```yaml
model:
  architecture: "yolo26n"
```

## 🧪 Testing

```bash
# Run training with minimal config for testing
python main.py train --training-config configs/test_config.yaml

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/models/list

# Test inference
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

## 📊 MLflow Features

### What's Logged

**Parameters**:
- Model architecture (yolo11n, yolo11s, etc.)
- All hyperparameters (epochs, batch size, lr, etc.)
- Data augmentation settings
- Dataset split ratios
- Seed for reproducibility
- GPU info

**Metrics (per epoch)**:
- mAP@50, mAP@50-95
- Precision, Recall, F1
- Box loss, class loss, DFL loss
- Training and validation metrics

**Artifacts**:
- Model weights (best.pt, last.pt)
- Confusion matrix
- PR curves, F1 curves
- Training curves (results.png)
- Validation prediction samples
- results.csv with all metrics

**Model Registry**:
- Automatic model registration
- Version control
- Stage management (Staging → Production)
- Model metadata and tags

## 🐛 Troubleshooting

### MLflow UI not accessible
```bash
# Check if services are running
docker-compose ps

# Restart MLflow
docker-compose restart mlflow

# Check logs
docker-compose logs mlflow
```

### Training fails with CUDA error
Edit `training_config.yaml`:
```yaml
device:
  device: "cpu"  # Force CPU if GPU issues
```

### No model in serving
```bash
# Check if model is registered
curl http://localhost:8000/models/list

# Manually load a model
curl -X POST "http://localhost:8000/models/load?model_name=yolov11-finger-counting&stage=Staging"
```

### Webcam not working
- Ensure HTTPS or localhost (browsers require secure context)
- Check browser permissions
- Try different browser (Chrome/Firefox recommended)

## 📝 License

[Your License Here]

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
- Open a GitHub issue
- Check MLflow UI for experiment details
- Review logs: `docker-compose logs <service-name>`

---

**Built with ❤️ using YOLOv11, MLflow, FastAPI, and Docker**

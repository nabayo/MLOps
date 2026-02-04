# MLOps Project - Technical Report

**Project**: End-to-End MLOps Platform for Finger Counting Detection
**Date**: February 2026
**Technologies**: YOLOv11, MLflow, FastAPI, Docker, PostgreSQL, MinIO

---

## Executive Summary

This project implements a complete MLOps platform for training, tracking, and deploying YOLOv11 object detection models for finger counting. The platform provides:

- **Automated training pipeline** with experiment tracking
- **Model versioning** and registry management
- **REST API** for real-time inference
- **Web dashboard** for monitoring and testing
- **Cross-platform scripts** for all operations
- **Backup/restore** capabilities for MLflow data

---

## 1. Project Architecture

### System Components

```
┌────────────────────────────────────────────────────────────────┐
│                   MLOps Platform                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────┐    ┌──────────────┐      ┌──────────────┐    │
│  │   Training   │──▶│   MLflow     │◀────│   Serving    │    │
│  │   Pipeline   │    │   Tracking   │      │     API      │    │
│  └──────────────┘    └──────────────┘      └──────────────┘    │
│         │                    │                    │            │
│         │                    │                    │            │
│         ▼                    ▼                    ▼            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Picsellia   │    │  PostgreSQL  │    │    Web       │      │
│  │   Dataset    │    │    MinIO     │    │  Dashboard   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Infrastructure Services

1. **MLflow** (Port 5000)
   - Experiment tracking
   - Model registry
   - Artifact storage coordination

2. **PostgreSQL**
   - MLflow metadata storage
   - Run parameters and metrics

3. **MinIO** (S3-compatible)
   - Model artifacts storage
   - Training outputs and logs

4. **Serving API** (Port 8000)
   - FastAPI-based inference endpoint
   - Model loading from MLflow
   - Preprocessing pipeline

5. **Web Dashboard** (Port 8080)
   - Live inference testing
   - Model selection UI
   - Results visualization

---

## 2. Dataset & Data Pipeline

### Dataset Source

- **Origin**: Combination of custom photos and [Roboflow Finger Count Dataset](https://universe.roboflow.com/finger-count/finger-count/)
- **Platform**: Picsellia for dataset management
- **Total Images**: 315 images
- **Classes**: 7 classes (0, 1, 2, 3, 4, 5, unknown)

### Data Preparation Pipeline

**Location**: `src/data_preparation.py`

**Process**:
1. **Download** from Picsellia API
2. **Convert** from Picsellia format to YOLO format
   - Parses UUID-keyed annotations
   - Converts bounding boxes (x, y, w, h) to normalized coordinates
   - Filters ACCEPTED annotations only
3. **Split** dataset: 80% train, 10% validation, 10% test
4. **Organize** into YOLO directory structure
5. **Generate** `data.yaml` for Ultralytics

**Key Features**:
- Automatic format conversion
- Reproducible splits (seeded random)
- Validation of image dimensions
- Error handling for corrupt annotations

---

## 3. Training Pipeline

### Training Architecture

**Location**: `src/training.py`, `main.py`

**Supported Models**:
- YOLOv11n (nano)
- YOLOv11s (small)
- YOLOv11m (medium)
- YOLOv11l (large)
- YOLOv11x (extra-large)

### MLflow Integration

**Automatic Logging**:
- All hyperparameters
- Training metrics (loss, precision, recall, mAP)
- Model artifacts
- Training curves and plots
- Configuration files

**Model Registry**:
- Automatic model registration
- Version management
- Stage transitions (None → Staging → Production)

### Training Configuration

**File**: `configs/training_config.yaml`

**Key Parameters**:
```yaml
training:
  epochs: 100
  batch_size: 16
  img_size: 640
  optimizer: auto
  device: cpu  # or cuda
  patience: 100

dataset:
  split_ratios:
    train: 0.8
    val: 0.1
    test: 0.1
  seed: 42
```

---

## 4. Evaluation & Metrics

**Location**: `src/evaluation.py`

**Metrics Computed**:
- **Per-class metrics**: Precision, Recall, F1-score
- **Overall metrics**: mAP@0.5, mAP@0.5:0.95
- **Box metrics**: Box loss, DFL loss, Class loss
- **Confusion matrix** for multi-class analysis

**Test Set Evaluation**:
- Automatic evaluation after training
- Results logged to MLflow
- Detailed class-wise performance

---

## 5. Model Serving

### REST API

**Location**: `serving/api.py`

**Endpoints**:
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Image inference
- `GET /metrics` - Model performance metrics

**Features**:
- Model caching for performance
- Automatic preprocessing (face blurring)
- Configurable confidence thresholds
- CORS enabled for web dashboard

### Preprocessing Pipeline

**Location**: `src/preprocessing/`

**Face Blurring** (`face_blur.py`):
- Automatic face detection using Haar Cascades
- Gaussian blur application
- Privacy protection in predictions

**Configurable via ENV**:
```bash
ENABLE_PREPROCESSING=true
```

---

## 6. Web Dashboard

**Location**: `frontend/`

**Features**:
- Drag-and-drop image upload
- Live inference results
- Model selection dropdown
- Bounding box visualization
- Class labels with confidence scores

**Technologies**:
- Vanilla JavaScript
- Modern CSS with gradients
- Responsive design
- Real-time API integration

---

## 7. Docker Infrastructure

### Docker Compose Services

**File**: `compose.yml`

**Services**:
1. **postgres** - Database for MLflow
2. **minio** - S3-compatible artifact storage
3. **minio-setup** - Bucket initialization
4. **mlflow** - Tracking server
5. **training** - Model training container
6. **serving** - Inference API
7. **frontend** - Web dashboard
8. **export** - Backup utility
9. **import** - Restore utility

### Health Checks

All services include health checks for:
- Service availability
- Database connectivity
- Storage accessibility
- Proper startup ordering

---

## 8. Automation & Scripts

### Cross-Platform Scripts

Created for **Linux/Mac** (.sh), **Windows PowerShell** (.ps1), and **Windows CMD** (.bat):

**Training**:
- `scripts/train.sh` - Run model training
- Automatic service startup
- Progress monitoring

**Serving**:
- `scripts/serve.sh` - Start API server
- Infrastructure validation
- Endpoint information

**Backup**:
- `scripts/docker_export.sh` - Export MLflow data to zip
- Timestamped backups
- Complete experiment preservation

**Restore**:
- `scripts/docker_import.sh` - Import MLflow data
- Collision handling (skip existing by default)
- Dry-run mode

---

## 9. Key Technical Achievements

### 1. Fixed Picsellia Data Parsing
**Problem**: Original code expected COCO format, but Picsellia uses UUID-keyed dictionaries
**Solution**: Completely rewrote parser in `data_preparation.py` to handle Picsellia's native format

### 2. Python 3.12 Compatibility
**Problem**: MLflow 2.10.2 incompatible with Python 3.12
**Solution**: Upgraded to MLflow 2.18.0 and added setuptools

### 3. Docker Build Optimization
**Problem**: Missing OpenCV dependencies, no layer caching
**Solution**: Added system dependencies, optimized Dockerfile with proper layer ordering

### 4. MLflow Backup/Restore System
**Innovation**: Complete export/import scripts for MLflow data
- Exports experiments, runs, models, and artifacts
- Handles collisions intelligently
- Cross-platform support

### 5. Cross-Platform Compatibility
**Achievement**: Full Windows support with PowerShell and Batch scripts
- Identical functionality across platforms
- Comprehensive documentation
- Error handling and validation

---

## 10. Project Structure

```
MLOps/
├── configs/
│   ├── config.yaml              # Dataset configuration
│   └── training_config.yaml     # Training hyperparameters
├── src/
│   ├── dataset_loader.py        # Picsellia dataset loading
│   ├── data_preparation.py      # YOLO format conversion
│   ├── training.py              # Training pipeline with MLflow
│   ├── evaluation.py            # Model evaluation metrics
│   ├── picsellia.py             # Picsellia API integration
│   └── preprocessing/
│       ├── base.py              # Preprocessing interface
│       └── face_blur.py         # Face detection & blurring
├── serving/
│   ├── api.py                   # FastAPI inference server
│   └── Dockerfile               # Serving container
├── frontend/
│   ├── index.html               # Web dashboard UI
│   ├── app.js                   # Dashboard logic
│   ├── style.css                # Modern styling
│   └── Dockerfile               # Frontend container
├── mlflow/
│   └── Dockerfile               # MLflow server container
├── scripts/
│   ├── train.sh/.ps1/.bat       # Training scripts
│   ├── serve.sh/.ps1/.bat       # Serving scripts
│   ├── docker_export.sh/.ps1/.bat # Backup scripts
│   ├── docker_import.sh/.ps1/.bat # Restore scripts
│   ├── export_mlflow.py         # Python export logic
│   ├── import_mlflow.py         # Python import logic
│   ├── WINDOWS_README.md        # Windows guide
│   ├── DOCKER_BACKUP.md         # Backup quick reference
│   └── BACKUP_README.md         # Full backup documentation
├── main.py                      # CLI entry point
├── Dockerfile                   # Training container
├── compose.yml                  # Docker Compose configuration
└── Readme.md                    # Main documentation
```

---

## 11. Configuration Management

### Environment Variables

**File**: `.env.example`

**Key Variables**:
```bash
# MLflow Configuration
MLFLOW_BACKEND_STORE_URI=postgresql+psycopg2://mlflow:mlflow@postgres:5432/mlflow
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# Model Configuration
DEFAULT_MODEL_NAME=yolo11-finger-counter
DEFAULT_MODEL_STAGE=Production

# Preprocessing
ENABLE_PREPROCESSING=true
```

---

## 12. Documentation

### Created Documentation

1. **README.md** - Main project documentation
2. **TRAINING.md** - Training guide
3. **DEPLOYMENT.md** - Deployment instructions
4. **scripts/WINDOWS_README.md** - Windows-specific guide
5. **scripts/DOCKER_BACKUP.md** - Backup quick reference
6. **scripts/BACKUP_README.md** - Comprehensive backup guide
7. **scripts/README.md** - Script documentation

---

## 13. Testing & Validation

### Successful Tests

✅ **Data Pipeline**:
- Picsellia dataset download
- Format conversion (315 images)
- YOLO format validation

✅ **Training**:
- YOLOv11 training execution
- MLflow logging
- Artifact storage

✅ **Serving**:
- API health checks
- Inference endpoint
- Model loading

✅ **Docker**:
- All services start successfully
- Health checks pass
- Network connectivity

✅ **Backup/Restore**:
- Export creates valid zip files
- Import correctly handles collisions
- Dry-run mode works

---

## 14. Future Improvements

### Potential Enhancements

1. **GPU Support**
   - CUDA-enabled Docker images
   - GPU resource allocation

2. **Advanced Monitoring**
   - Prometheus metrics
   - Grafana dashboards
   - Alert system

3. **CI/CD Pipeline**
   - Automated testing
   - Model deployment automation
   - Version control integration

4. **Model Optimization**
   - Model quantization
   - TensorRT acceleration
   - ONNX export

5. **Enhanced UI**
   - Real-time video inference
   - Model comparison tools
   - Training progress visualization

---

## 15. Challenges Overcome

### Technical Challenges

1. **Picsellia Data Format**
   - Challenge: Undocumented UUID-based annotation format
   - Solution: Reverse-engineered format from JSON structure
   - Result: Robust parser handling 405 annotations

2. **Docker Build Issues**
   - Challenge: Missing system dependencies for OpenCV
   - Solution: Updated package names for Debian Bookworm
   - Result: Clean builds on Python 3.12-slim

3. **MLflow Python 3.12**
   - Challenge: Incompatible importlib.metadata
   - Solution: Upgraded to MLflow 2.18.0
   - Result: Full Python 3.12 compatibility

4. **Cross-Platform Scripts**
   - Challenge: Different shell syntaxes (bash, PowerShell, cmd)
   - Solution: Created equivalent scripts for all platforms
   - Result: Identical functionality across OS

5. **MLflow Data Portability**
   - Challenge: No built-in export/import
   - Solution: Custom Python scripts with full metadata preservation
   - Result: Complete backup/restore capability

---

## 16. Technologies Used

### Core Technologies

- **Python 3.12** - Main programming language
- **YOLOv11** - Object detection model
- **Ultralytics** - YOLO framework
- **MLflow 2.18.0** - Experiment tracking
- **FastAPI** - REST API framework
- **Docker & Docker Compose** - Containerization

### Infrastructure

- **PostgreSQL** - MLflow metadata storage
- **MinIO** - S3-compatible object storage
- **Nginx** - Frontend web server

### Libraries

- **OpenCV** - Image preprocessing
- **Pillow** - Image manipulation
- **NumPy** - Numerical operations
- **PyYAML** - Configuration management
- **Picsellia SDK** - Dataset management

---

## 17. Deployment

### Local Deployment

```bash
# 1. Start all services
docker-compose up -d

# 2. Run training
./scripts/train.sh

# 3. Start serving API
./scripts/serve.sh

# 4. Access web dashboard
open http://localhost:8080
```

### Production Considerations

- Environment variable management
- SSL/TLS termination
- Authentication & authorization
- Resource limits & scaling
- Backup automation
- Monitoring & alerting

---

## 18. Performance Metrics

### Training Performance

- **YOLOv11n**: ~10-15 min/epoch (CPU)
- **Dataset**: 315 images processed
- **Classes**: 7 detected classes
- **Valid Images**: ~285 (30 with annotation errors)

### API Performance

- **Inference**: ~200-500ms per image (CPU)
- **Model Loading**: ~2-3s (cached after first load)
- **API Startup**: ~5-10s

### Storage Requirements

- **Docker Images**: ~2-3 GB total
- **Model Artifacts**: ~20-30 MB per model
- **Experiment Data**: ~50-100 MB per training run

---

## 19. Security Considerations

### Implemented Security

1. **Face Blurring** - Privacy protection in predictions
2. **Read-only Mounts** - Sensitive files mounted as read-only
3. **Environment Variables** - Secrets managed via .env
4. **Network Isolation** - Docker network segmentation
5. **CORS Configuration** - Controlled API access

### Recommended Additions

- API authentication (JWT tokens)
- HTTPS/TLS encryption
- Rate limiting
- Input validation & sanitization
- Secrets management (Vault, AWS Secrets Manager)

---

## 20. Conclusion

This MLOps project successfully implements a complete end-to-end pipeline for object detection model training and deployment. Key achievements include:

✅ **Complete Infrastructure** - All services containerized and orchestrated
✅ **Automated Pipeline** - From data preparation to model deployment
✅ **Experiment Tracking** - Full MLflow integration with model registry
✅ **Production-Ready API** - FastAPI serving with preprocessing
✅ **User-Friendly Interface** - Web dashboard for testing
✅ **Cross-Platform Support** - Scripts for Linux, Mac, and Windows
✅ **Backup/Restore** - Complete data portability
✅ **Comprehensive Documentation** - Multiple guides for different use cases

The platform is production-ready and can be extended for various object detection tasks beyond finger counting.

---

## Appendix A: Quick Start Commands

```bash
# Training
./scripts/train.sh

# Serving only
./scripts/serve.sh

# Export MLflow data
./scripts/docker_export.sh backup_name

# Import MLflow data
./scripts/docker_import.sh backup_name.zip

# View logs
docker-compose logs -f mlflow
docker-compose logs -f serving

# Stop services
docker-compose stop
```

## Appendix B: Useful URLs

- **MLflow UI**: http://localhost:5000
- **API Documentation**: http://localhost:8000/docs
- **Web Dashboard**: http://localhost:8080
- **MinIO Console**: http://localhost:9001

---

**End of Report**

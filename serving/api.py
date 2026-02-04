"""
FastAPI Serving Application with MLflow Model Registry Integration.

Features:
- Dynamic model loading from MLflow
- Live model switching without restart
- Experiments and metrics browser
- Configurable preprocessing pipeline
- Real-time inference with finger counting
"""

import os
import io
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv11 Finger Counting Serving API",
    description="MLOps serving API with dynamic model loading and MLflow integration",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Global model state
current_model = None
current_model_info = {}
preprocessing_enabled = os.getenv('ENABLE_PREPROCESSING', 'false').lower() == 'true'


# Pydantic models
class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    version: Optional[str]
    stage: Optional[str]
    architecture: Optional[str]
    metrics: Dict[str, float]
    loaded_at: str


class PredictionBox(BaseModel):
    """Single prediction bounding box."""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]


class PredictionResponse(BaseModel):
    """Inference prediction response."""
    finger_count: int
    predictions: List[PredictionBox]
    preprocessing_applied: bool
    inference_time_ms: float


class ExperimentInfo(BaseModel):
    """Experiment information."""
    experiment_id: str
    experiment_name: str
    run_count: int


class RunInfo(BaseModel):
    """Run information."""
    run_id: str
    run_name: str
    experiment_id: str
    start_time: str
    status: str
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


# Helper functions
def load_model_from_registry(
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Registered model name
        version: Model version (e.g., "1", "2")
        stage: Model stage ("Staging", "Production", "Archived")

    Returns:
        Model info dictionary
    """
    global current_model, current_model_info

    try:
        # Determine model URI
        if version:
            model_uri = f"models:/{model_name}/{version}"
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
        else:
            # Default to Production
            model_uri = f"models:/{model_name}/Production"

        print(f"Loading model from: {model_uri}")

        # Get model version details
        if version:
            model_version = client.get_model_version(model_name, version)
        else:
            # Get latest version in stage
            versions = client.get_latest_versions(model_name, stages=[stage] if stage else ["Production"])
            if not versions:
                raise ValueError(f"No model found for {model_name} in stage {stage}")
            model_version = versions[0]

        # Download model artifacts and load
        # Note: mlflow.pytorch.load_model doesn't work well with YOLO
        # We need to download the weights file directly
        run_id = model_version.run_id
        run = client.get_run(run_id)

        # Try to find weights in artifacts
        artifacts = client.list_artifacts(run_id, "weights")
        best_weights_path = None

        for artifact in artifacts:
            if artifact.path.endswith("best.pt"):
                best_weights_path = artifact.path
                break

        if not best_weights_path:
            raise ValueError("Could not find best.pt in model artifacts")

        # Download artifact
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=best_weights_path
        )

        # Load YOLO model
        model = YOLO(local_path)

        # Get metrics
        metrics = {}
        for key, value in run.data.metrics.items():
            if 'final_' in key:
                metrics[key.replace('final_', '')] = value

        # Get parameters
        params = run.data.params

        # Store model info
        model_info = {
            'name': model_name,
            'version': model_version.version,
            'stage': model_version.current_stage,
            'architecture': params.get('model_architecture', 'unknown'),
            'metrics': metrics,
            'run_id': run_id,
            'loaded_at': datetime.now().isoformat()
        }

        current_model = model
        current_model_info = model_info

        print(f"✓ Model loaded: {model_name} v{model_version.version} ({model_version.current_stage})")
        return model_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing pipeline to image.

    Args:
        image: Input image in BGR format

    Returns:
        Preprocessed image
    """
    if not preprocessing_enabled:
        return image

    # For now, use fast face blur with OpenCV (deface is too slow for API)
    from src.preprocessing import FastFaceBlurStep

    blur_step = FastFaceBlurStep(blur_kernel_size=51)
    return blur_step.process(image)


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    model_name = os.getenv('MODEL_NAME', 'yolov11-finger-counting')
    model_stage = os.getenv('MODEL_STAGE', 'Production')

    try:
        load_model_from_registry(model_name, stage=model_stage)
        print(f"✓ Default model loaded: {model_name} ({model_stage})")
    except Exception as e:
        print(f"⚠ Could not load default model: {e}")
        print("  API will start without a loaded model")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_loaded": current_model is not None,
        "preprocessing_enabled": preprocessing_enabled
    }


@app.get("/models/current", response_model=ModelInfo)
async def get_current_model():
    """Get information about currently loaded model."""
    if current_model is None:
        raise HTTPException(status_code=404, detail="No model currently loaded")

    return ModelInfo(**current_model_info)


@app.get("/models/list")
async def list_models() -> List[Dict[str, Any]]:
    """List all registered models from MLflow."""
    try:
        registered_models = client.search_registered_models()

        models_info = []
        for rm in registered_models:
            # Get latest versions
            latest_versions = client.get_latest_versions(rm.name)

            versions = []
            for mv in latest_versions:
                # Get run metrics
                run = client.get_run(mv.run_id)
                metrics = {k.replace('final_', ''): v for k, v in run.data.metrics.items() if 'final_' in k}

                versions.append({
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'run_id': mv.run_id,
                    'metrics': metrics,
                    'created_at': datetime.fromtimestamp(mv.creation_timestamp / 1000).isoformat()
                })

            models_info.append({
                'name': rm.name,
                'description': rm.description or '',
                'versions': versions
            })

        return models_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@app.post("/models/load")
async def load_model(
    model_name: str = Query(..., description="Model name"),
    version: Optional[str] = Query(None, description="Model version"),
    stage: Optional[str] = Query(None, description="Model stage")
) -> Dict[str, Any]:
    """Dynamically load a model from MLflow registry."""
    model_info = load_model_from_registry(model_name, version, stage)
    return {
        "status": "success",
        "message": f"Model {model_name} loaded successfully",
        "model_info": model_info
    }


@app.get("/models/experiments")
async def list_experiments() -> List[Dict[str, Any]]:
    """List all MLflow experiments with runs."""
    try:
        experiments = client.search_experiments()

        experiments_info = []
        for exp in experiments:
            # Get runs for this experiment
            runs = client.search_runs(exp.experiment_id, max_results=100)

            runs_info = []
            for run in runs:
                # Extract metrics
                metrics = {}
                for key, value in run.data.metrics.items():
                    if 'final_' in key:
                        metrics[key.replace('final_', '')] = value

                runs_info.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'start_time': datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                    'status': run.info.status,
                    'metrics': metrics,
                    'params': dict(run.data.params),
                    'tags': dict(run.data.tags)
                })

            experiments_info.append({
                'experiment_id': exp.experiment_id,
                'experiment_name': exp.name,
                'run_count': len(runs),
                'runs': runs_info
            })

        return experiments_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list experiments: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Run inference on uploaded image.

    Returns finger count and predictions.
    """
    if current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocessing
        preprocessing_applied = False
        if preprocessing_enabled:
            image = preprocess_image(image)
            preprocessing_applied = True

        # Run inference
        import time
        start_time = time.time()

        results = current_model.predict(
            image,
            conf=0.25,
            iou=0.7,
            verbose=False
        )

        inference_time = (time.time() - start_time) * 1000

        # Parse results
        predictions = []
        finger_count = 0

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes[i]

                    # Get class info
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])

                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    predictions.append(PredictionBox(
                        class_name=class_name,
                        class_id=class_id,
                        confidence=confidence,
                        bbox=[x1, y1, x2, y2]
                    ))

                    # Count fingers
                    # Assuming class names like "1", "2", "3", "4", "5" or "finger-1", etc.
                    try:
                        if class_name.isdigit():
                            finger_count += int(class_name)
                        elif '-' in class_name:
                            finger_count += int(class_name.split('-')[-1])
                    except:
                        finger_count += 1  # Fallback

        return PredictionResponse(
            finger_count=finger_count,
            predictions=predictions,
            preprocessing_applied=preprocessing_applied,
            inference_time_ms=round(inference_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get simple metrics (for Prometheus, etc.)."""
    return {
        "model_loaded": current_model is not None,
        "model_info": current_model_info if current_model else {},
        "preprocessing_enabled": preprocessing_enabled,
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

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
import base64
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
    processed_image: Optional[str] = None  # Base64 encoded processed image


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
        # The model source tells us exactly where the weights are
        run_id = model_version.run_id
        run = client.get_run(run_id)

        # Parse the source URI to get the artifact path
        # Source format: "runs:/<run_id>/<artifact_path>"
        source = model_version.source
        print(f"Model source: {source}")
        
        if source.startswith("runs:/"):
            # Extract artifact path from source
            parts = source.replace("runs:/", "").split("/", 1)
            if len(parts) == 2:
                artifact_path = parts[1]  # e.g., "weights/best.pt"
            else:
                # Fallback to default path
                artifact_path = "weights/best.pt"
        else:
            # Direct S3 path or other format
            artifact_path = "weights/best.pt"
        
        print(f"Attempting to download: {artifact_path}")

        # Download artifact directly (don't list, just download)
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path
            )
            print(f"✓ Downloaded to: {local_path}")
        except Exception as download_error:
            # If the exact path fails, try common alternatives
            print(f"First attempt failed, trying alternatives...")
            alternatives = [
                "weights/best.pt",
                "model/weights/best.pt", 
                "best.pt"
            ]
            
            local_path = None
            for alt_path in alternatives:
                try:
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=alt_path
                    )
                    print(f"✓ Found at: {alt_path}")
                    break
                except:
                    continue
            
            if not local_path:
                raise ValueError(f"Could not find model weights. Tried: {artifact_path}, {alternatives}")

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
    try:
        # Add parent directory to path if not already there
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        from src.preprocessing import FastFaceBlurStep
        blur_step = FastFaceBlurStep(blur_kernel_size=51)
        return blur_step.process(image)
    except ImportError as e:
        print(f"Warning: Could not import preprocessing module: {e}")
        # Fallback to simple Gaussian blur if module not found
        return cv2.GaussianBlur(image, (51, 51), 0)


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
        
        # Fallback: Check if default model exists (search sometimes fails)
        default_model_name = os.getenv('MODEL_NAME', 'yolov11-finger-counting')
        
        # Normalize names for comparison
        found_names = [rm.name for rm in registered_models]
        
        # Try both the env var name and the Capitalized version we likely created
        for name_to_check in [default_model_name, "YOLOv11-Finger-Counter"]:
            if name_to_check not in found_names:
                try:
                    m = client.get_registered_model(name_to_check)
                    print(f"Fallback: Found {name_to_check} via direct lookup")
                    registered_models.append(m)
                except:
                    pass


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
            print(f"🔄 Preprocessing enabled, applying face blur...")
            try:
                image = preprocess_image(image)
                preprocessing_applied = True
                print(f"✓ Preprocessing applied successfully")
            except Exception as prep_error:
                print(f"⚠ Preprocessing failed: {prep_error}")
                # Continue without preprocessing
                import traceback
                traceback.print_exc()

        # Run inference
        import time
        start_time = time.time()

        try:
            results = current_model.predict(
                image,
                conf=0.25,
                iou=0.7,
                verbose=False
            )
        except Exception as inf_error:
            print(f"❌ Inference failed: {inf_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(inf_error)}")

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

        print(f"✓ Inference complete: {len(predictions)} predictions, finger_count={finger_count}, time={inference_time:.2f}ms")

        # Encode processed image as base64 for frontend display
        processed_image_b64 = None
        if preprocessing_applied:
            # Encode the processed image
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return PredictionResponse(
            finger_count=finger_count,
            predictions=predictions,
            preprocessing_applied=preprocessing_applied,
            inference_time_ms=round(inference_time, 2),
            processed_image=processed_image_b64
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
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

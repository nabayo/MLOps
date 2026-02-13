"""
FastAPI Serving Application with MLflow Model Registry Integration.

Features:
- Dynamic model loading from MLflow
- Live model switching without restart
- Experiments and metrics browser
- Configurable preprocessing pipeline
- Real-time inference with finger counting
"""

from typing import Any, Optional
from datetime import datetime
from pathlib import Path

import os
import sys
import time

import traceback
import mlflow
import tempfile
import uvicorn
import numpy as np

import cv2

from ultralytics import YOLO
from pydantic import BaseModel
from dotenv import load_dotenv

from mlflow.tracking import MlflowClient

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

try:
    # Add parent directory to path if not already there
    PARENT_DIR = str(Path(__file__).parent.parent)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)

    from src.preprocessing import FastFaceBlurStep

except ImportError as e:
    print(f"Warning: Could not import preprocessing module: {e}")
    FastFaceBlurStep = None

# Load environment
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="YOLO Finger Counting Serving API",
    description="MLOps serving API with dynamic model loading and MLflow integration",
    version="1.0.0",
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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# Global configuration
preprocessing_enabled = os.getenv("ENABLE_PREPROCESSING", "false").lower() == "true"


# Pydantic models
class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    version: Optional[str]
    stage: Optional[str]
    architecture: Optional[str]
    metrics: dict[str, float]
    loaded_at: str


class SkipFrameConfig(BaseModel):
    """Configuration for frame skipping."""

    skip_frames: int


class PredictionBox(BaseModel):
    """Single prediction bounding box."""

    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]


class PredictionResponse(BaseModel):
    """Inference prediction response."""

    finger_count: int
    predictions: list[PredictionBox]
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
    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]
    available_weights: list[str] = []


class GlobalState:
    """Manages global application state without using the 'global' statement."""

    def __init__(self):
        self.current_model = None
        self.current_model_info: dict[str, Any] = {}
        self.blur_step: Any = None
        self.skip_frame_limit = 6
        self.frame_counter = 0
        self.last_prediction: Optional[PredictionResponse] = None
        self.ws_conf_threshold: float = 0.15
        # Experiments cache
        self.experiments_cache: Optional[list[dict[str, Any]]] = None
        self.experiments_cache_time: float = 0.0


state = GlobalState()


# Helper functions
def load_model_from_registry(
    model_name: str, version: Optional[str] = None, stage: Optional[str] = None
) -> dict[str, Any]:
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Registered model name
        version: Model version (e.g., "1", "2")
        stage: Model stage ("Staging", "Production", "Archived")

    Returns:
        Model info dictionary
    """

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
            versions = client.get_latest_versions(
                model_name, stages=[stage] if stage else ["Production"]
            )
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
                run_id=run_id, artifact_path=artifact_path
            )
            print(f"✓ Downloaded to: {local_path}")
        except Exception as _download_error:  # pylint: disable=broad-except
            # If the exact path fails, try common alternatives
            print("First attempt failed, trying alternatives...")
            alternatives = ["weights/best.pt", "model/weights/best.pt", "best.pt"]

            local_path = None
            for alt_path in alternatives:
                try:
                    local_path = mlflow.artifacts.download_artifacts(
                        run_id=run_id, artifact_path=alt_path
                    )
                    print(f"✓ Found at: {alt_path}")
                    break
                except Exception as _e:  # pylint: disable=broad-except
                    continue

            if not local_path:
                raise ValueError(
                    f"Could not find model weights.\
                         Tried: {artifact_path}, {alternatives}"
                ) from _download_error

        # Load YOLO model
        model = YOLO(local_path)

        # Get metrics
        metrics = {}
        for key, value in run.data.metrics.items():
            if "final_" in key:
                metrics[key.replace("final_", "")] = value

        # Get parameters
        params = run.data.params

        # Store model info
        model_info = {
            "name": model_name,
            "version": model_version.version,
            "stage": model_version.current_stage,
            "architecture": params.get("model_architecture", "unknown"),
            "metrics": metrics,
            "run_id": run_id,
            "loaded_at": datetime.now().isoformat(),
        }

        state.current_model = model
        state.current_model_info = model_info

        print(
            f"✓ Model loaded: {model_name} v{model_version.version} ({model_version.current_stage})"
        )
        return model_info

    except Exception as e:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500, detail=f"Failed to load model: {str(e)}"
        ) from e


def get_blur_step():
    """Lazy initialization of blur step."""
    if state.blur_step is None:
        try:
            print("Initializing FastFaceBlurStep...")
            state.blur_step = FastFaceBlurStep(blur_kernel_size=51)
        except ImportError as e:
            print(f"Warning: Could not import preprocessing module: {e}")
            return None
    return state.blur_step


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

    # Use global/cached step instance
    step = get_blur_step()

    if step:
        return step.process(image)
    else:
        # Fallback to simple Gaussian blur if module not found
        return cv2.GaussianBlur(image, (51, 51), 0)  # pylint: disable=no-member


# API Endpoints


@app.on_event("startup")
async def startup_event():
    """Load default model and preprocessing on startup."""

    # Initialize preprocessing if enabled
    if preprocessing_enabled:
        print("Pre-initializing face blur model...")
        get_blur_step()

    model_name = os.getenv("MODEL_NAME", "yolov11-finger-counting")
    model_stage = os.getenv("MODEL_STAGE", "Production")

    try:
        load_model_from_registry(model_name, stage=model_stage)
        print(f"✓ Default model loaded: {model_name} ({model_stage})")
    except Exception as e:  # pylint: disable=broad-except
        print(f"⚠ Could not load default model: {e}")
        print("  API will start without a loaded model")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_loaded": state.current_model is not None,
        "preprocessing_enabled": preprocessing_enabled,
    }


@app.get("/models/current", response_model=ModelInfo)
async def get_current_model():
    """Get information about currently loaded model."""
    if state.current_model is None:
        raise HTTPException(status_code=404, detail="No model currently loaded")

    return ModelInfo(**state.current_model_info)


@app.get("/models/list")
async def list_models() -> list[dict[str, Any]]:
    """List all registered models from MLflow."""
    try:
        registered_models = client.search_registered_models()

        # Fallback: Check if default model exists (search sometimes fails)
        default_model_name = os.getenv("MODEL_NAME", "")

        # Normalize names for comparison
        found_names = [rm.name for rm in registered_models]

        # Try both the env var name and the Capitalized version we likely created
        for name_to_check in [default_model_name, "YOLOv26", "YOLOv11-Finger-Counter"]:
            if name_to_check not in found_names:
                try:
                    m = client.get_registered_model(name_to_check)
                    print(f"Fallback: Found {name_to_check} via direct lookup")
                    registered_models.append(m)
                except Exception as _e:  # pylint: disable=broad-except
                    pass

        models_info = []
        for rm in registered_models:
            # Get latest versions
            latest_versions = client.get_latest_versions(rm.name)

            versions = []
            for mv in latest_versions:
                # Get run metrics
                run = client.get_run(mv.run_id)
                metrics = {
                    k.replace("final_", ""): v
                    for k, v in run.data.metrics.items()
                    if "final_" in k
                }

                versions.append(
                    {
                        "version": mv.version,
                        "stage": mv.current_stage,
                        "run_id": mv.run_id,
                        "metrics": metrics,
                        "created_at": datetime.fromtimestamp(
                            mv.creation_timestamp / 1000
                        ).isoformat(),
                    }
                )

            models_info.append(
                {
                    "name": rm.name,
                    "description": rm.description or "",
                    "versions": versions,
                }
            )

        return models_info

    except Exception as e:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500, detail=f"Failed to list models: {str(e)}"
        ) from e


@app.post("/models/load")
async def load_model(
    model_name: str = Query(..., description="Model name"),
    version: Optional[str] = Query(None, description="Model version"),
    stage: Optional[str] = Query(None, description="Model stage"),
) -> dict[str, Any]:
    """Dynamically load a model from MLflow registry."""
    model_info = load_model_from_registry(model_name, version, stage)
    return {
        "status": "success",
        "message": f"Model {model_name} loaded successfully",
        "model_info": model_info,
    }


def _fetch_experiments() -> list[dict[str, Any]]:
    """Fetch experiments from MLflow and update the cache."""
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
                if "final_" in key:
                    metrics[key.replace("final_", "")] = value

            runs_info.append(
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
                    "start_time": datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).isoformat(),
                    "status": run.info.status,
                    "metrics": metrics,
                    "params": dict(run.data.params),
                    "tags": dict(run.data.tags),
                    "available_weights": check_run_weights(run.info.run_id),
                }
            )

        experiments_info.append(
            {
                "experiment_id": exp.experiment_id,
                "experiment_name": exp.name,
                "run_count": len(runs),
                "runs": runs_info,
            }
        )

    state.experiments_cache = experiments_info
    state.experiments_cache_time = time.time()
    return experiments_info


@app.get("/models/experiments")
async def list_experiments() -> list[dict[str, Any]]:
    """List all MLflow experiments with runs (cached)."""
    try:
        # Return cached data if still valid
        if state.experiments_cache is not None:
            return state.experiments_cache

        return _fetch_experiments()

    except Exception as e:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list experiments: {str(e)}",
        ) from e


@app.post("/models/experiments/refresh")
async def refresh_experiments() -> dict[str, Any]:
    """Force-refresh the experiments cache."""
    try:
        data = _fetch_experiments()
        return {
            "status": "success",
            "experiment_count": len(data),
            "cached_at": datetime.fromtimestamp(
                state.experiments_cache_time
            ).isoformat(),
        }
    except Exception as e:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh experiments: {str(e)}",
        ) from e


@app.post("/config/skip_frames")
async def set_skip_frames(config: SkipFrameConfig):
    """Set the number of frames to skip between inferences."""
    state.skip_frame_limit = config.skip_frames
    state.frame_counter = 0  # Reset counter when config changes
    print(f"✓ Skip frame limit set to: {state.skip_frame_limit}")
    return {"status": "success", "skip_frames": state.skip_frame_limit}


@app.post("/config/conf_threshold")
async def set_conf_threshold(
    threshold: float = Query(
        ..., description="Confidence threshold (0.0–1.0)", ge=0.0, le=1.0
    ),
):
    """Set the confidence threshold for WebSocket predictions."""
    state.ws_conf_threshold = threshold
    print(f"✓ Confidence threshold set to: {state.ws_conf_threshold}")
    return {"status": "success", "conf_threshold": state.ws_conf_threshold}


def check_run_weights(run_id: str) -> list[str]:
    """
    Check for available weights for a specific run.
    Uses 'blind download' to verify existence.
    """
    verified_weights = []
    potential_weights = ["weights/best.pt", "weights/last.pt", "best.pt", "last.pt"]

    # We use a single temp directory for all checks to keep it clean
    with tempfile.TemporaryDirectory() as temp_dir:
        for weight_path in potential_weights:
            try:
                # Blind download attempt
                client.download_artifacts(run_id, weight_path, dst_path=temp_dir)
                verified_weights.append(weight_path)
            except Exception:  # pylint: disable=broad-except
                # Failed means likely doesn't exist
                pass

    return verified_weights


@app.post("/models/load_run_weights")
async def load_run_weights(
    run_id: str = Query(..., description="MLflow Run ID"),
    artifact_path: str = Query(
        ..., description="Path to weight file (e.g., weights/best.pt)"
    ),
):
    """
    Load a model from a specific run's weight file.
    """

    print(f"Loading model from run: {run_id}, path: {artifact_path}")

    try:
        run = client.get_run(run_id)

        # Download the specific artifact
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            local_path = client.download_artifacts(
                run_id, artifact_path, dst_path=os.path.dirname(tmp.name)
            )

        #
        cache_dir = os.path.join(os.getcwd(), "models_cache", run_id)
        os.makedirs(cache_dir, exist_ok=True)

        local_path = client.download_artifacts(
            run_id, artifact_path, dst_path=cache_dir
        )

        # If artifact_path is 'weights/best.pt', it's at cache_dir/weights/best.pt
        full_path = os.path.join(cache_dir, artifact_path)

        if not os.path.exists(full_path):
            # download_artifacts return value is the local path
            full_path = local_path

        print(f"✓ Downloaded to: {full_path}")

        # Load YOLO model
        model = YOLO(full_path)

        # Extract metrics
        metrics = {}
        for key, value in run.data.metrics.items():
            if "final_" in key:
                metrics[key.replace("final_", "")] = value

        # Update state
        model_info = {
            "name": f"Run {run_id[:8]}",
            "version": artifact_path,
            "stage": "Experiment",
            "architecture": run.data.params.get("model_architecture", "unknown"),
            "metrics": metrics,
            "run_id": run_id,
            "loaded_at": datetime.now().isoformat(),
        }

        state.current_model = model
        state.current_model_info = model_info

        return {
            "status": "success",
            "message": f"Loaded {artifact_path} from run {run_id}",
            "model_info": model_info,
        }

    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Load failed: {e}")

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Failed to load weights: {str(e)}"
        ) from e


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    skip_check: bool = Query(True, description="Whether to apply frame skipping logic"),
    conf_threshold: float = Query(
        0.15, description="Confidence threshold for predictions", ge=0.0, le=1.0
    ),
):
    """
    Run inference on uploaded image.

    Returns finger count and predictions.
    """
    if state.current_model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # pylint: disable=no-member

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Skip frame logic

        # Increment counter
        state.frame_counter += 1

        # Check if we should skip
        if skip_check and state.frame_counter < state.skip_frame_limit:
            # Skip inference
            # print(f"⏭ Skipping frame {state.frame_counter}/{state.skip_frame_limit}")

            # Return last prediction if available to keep UI stable
            if state.last_prediction:
                # Update inference time to 0 to indicate skip
                last_response = state.last_prediction.copy()
                last_response.inference_time_ms = 0
                return last_response

            # If no last prediction, return empty
            return PredictionResponse(
                finger_count=0,
                predictions=[],
                preprocessing_applied=False,
                inference_time_ms=0,
                processed_image=None,
            )

        # Start Frame processing, Reset counter
        state.frame_counter = 0

        # Preprocessing
        preprocessing_applied = False
        if preprocessing_enabled:
            print("🔄 Preprocessing enabled, applying face blur...")
            try:
                image = preprocess_image(image)
                preprocessing_applied = True
                print("✓ Preprocessing applied successfully")
            except Exception as prep_error:  # pylint: disable=broad-except
                print(f"⚠ Preprocessing failed: {prep_error}")
                # Continue without preprocessing
                traceback.print_exc()

        # Run inference

        start_time = time.time()

        try:
            results = state.current_model.predict(
                image, conf=conf_threshold, iou=0.7, verbose=False
            )
        except Exception as inf_error:
            print(f"❌ Inference failed: {inf_error}")

            traceback.print_exc()
            raise HTTPException(
                status_code=500, detail=f"Inference failed: {str(inf_error)}"
            ) from inf_error

        inference_time = (time.time() - start_time) * 1000

        # Parse results
        predictions = []
        finger_count = 0

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes

                for box in boxes:
                    # Get class info
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])

                    # Get bbox coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    predictions.append(
                        PredictionBox(
                            class_name=class_name,
                            class_id=class_id,
                            confidence=confidence,
                            bbox=[x1, y1, x2, y2],
                        )
                    )

                    # Count fingers
                    # Assuming class names like "1", "2", "3", "4", "5" or "finger-1", etc.
                    try:
                        if class_name.isdigit():
                            finger_count += int(class_name)
                        elif "-" in class_name:
                            finger_count += int(class_name.split("-")[-1])
                    except Exception as _e:  # pylint: disable=broad-except
                        finger_count += 1  # Fallback

        print(
            f"✓ Inference complete: {len(predictions)} predictions, finger_count={finger_count}, time={inference_time:.2f}ms"
        )

        response = PredictionResponse(
            finger_count=finger_count,
            predictions=predictions,
            preprocessing_applied=preprocessing_applied,
            inference_time_ms=round(inference_time, 2),
            processed_image=None,
        )

        # Update global last prediction
        state.last_prediction = response

        return response

    except HTTPException:
        raise
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Prediction error: {e}")

        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        ) from e


@app.get("/metrics")
async def get_metrics():
    """Get simple metrics (for Prometheus, etc.)."""
    return {
        "model_loaded": state.current_model is not None,
        "model_info": state.current_model_info if state.current_model else {},
        "preprocessing_enabled": preprocessing_enabled,
        "skip_frames": state.skip_frame_limit,
    }


@app.websocket("/ws/predict")
async def ws_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time inference.

    Protocol:
        Client sends: binary JPEG frame
        Server sends: JSON PredictionResponse
    """
    await websocket.accept()
    print("🔌 WebSocket client connected")

    try:
        while True:
            # Receive binary JPEG data from client
            data = await websocket.receive_bytes()

            if state.current_model is None:
                await websocket.send_json(
                    {"error": "No model loaded", "finger_count": 0, "predictions": []}
                )
                continue

            try:
                # Decode image
                nparr = np.frombuffer(data, np.uint8)
                image = cv2.imdecode(  # pylint: disable=no-member
                    nparr,
                    cv2.IMREAD_COLOR,  # pylint: disable=no-member
                )

                if image is None:
                    await websocket.send_json(
                        {"error": "Invalid image", "finger_count": 0, "predictions": []}
                    )
                    continue

                # Preprocessing
                preprocessing_applied = False
                if preprocessing_enabled:
                    try:
                        image = preprocess_image(image)
                        preprocessing_applied = True
                    except Exception:  # pylint: disable=broad-except
                        pass

                # Run inference
                start_time = time.time()
                results = state.current_model.predict(
                    image, conf=state.ws_conf_threshold, iou=0.7, verbose=False
                )
                inference_time = (time.time() - start_time) * 1000

                # Parse results
                predictions = []
                finger_count = 0

                if len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].tolist()

                            predictions.append(
                                {
                                    "class_name": class_name,
                                    "class_id": class_id,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2],
                                }
                            )

                            try:
                                if class_name.isdigit():
                                    finger_count += int(class_name)
                                elif "-" in class_name:
                                    finger_count += int(class_name.split("-")[-1])
                            except Exception:  # pylint: disable=broad-except
                                finger_count += 1

                await websocket.send_json(
                    {
                        "finger_count": finger_count,
                        "predictions": predictions,
                        "preprocessing_applied": preprocessing_applied,
                        "inference_time_ms": round(inference_time, 2),
                    }
                )

            except Exception as e:  # pylint: disable=broad-except
                print(f"❌ WebSocket inference error: {e}")
                await websocket.send_json(
                    {
                        "error": str(e),
                        "finger_count": 0,
                        "predictions": [],
                        "preprocessing_applied": False,
                        "inference_time_ms": 0,
                    }
                )

    except WebSocketDisconnect:
        print("🔌 WebSocket client disconnected")
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ WebSocket connection error: {e}")


if __name__ == "__main__":
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

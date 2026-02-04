import mlflow
from mlflow.tracking import MlflowClient
import os

# Setup
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()

model_name = "yolov11-finger-counting"
version = "1"

print(f"Testing load for {model_name} version {version}")

try:
    # Get model version
    model_version = client.get_model_version(model_name, version)
    print(f"✓ Found model version: {model_version.version}")
    print(f"  Stage: {model_version.current_stage}")
    print(f"  Run ID: {model_version.run_id}")
    print(f"  Source: {model_version.source}")
    
    # Get run
    run_id = model_version.run_id
    run = client.get_run(run_id)
    print(f"✓ Found run")
    
    # List artifacts
    print("\nListing artifacts in 'weights':")
    try:
        artifacts = client.list_artifacts(run_id, "weights")
        for art in artifacts:
            print(f"  - {art.path} (is_dir: {art.is_dir}, size: {art.file_size})")
    except Exception as e:
        print(f"  Error: {e}")
        
    # Try root artifacts
    print("\nListing root artifacts:")
    try:
        artifacts = client.list_artifacts(run_id)
        for art in artifacts:
            print(f"  - {art.path} (is_dir: {art.is_dir})")
    except Exception as e:
        print(f"  Error: {e}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

import mlflow
from mlflow.tracking import MlflowClient
import os

os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')
client = MlflowClient()

model_name = "yolov11-finger-counting"
version = "1"

print(f"Checking artifacts for {model_name} version {version}\n")

model_version = client.get_model_version(model_name, version)
run_id = model_version.run_id

print(f"Run ID: {run_id}")
print(f"Source URI: {model_version.source}\n")

# The source tells us where MLflow expects to find the model
# Let's parse it
source = model_version.source
print(f"Model source: {source}")

#Try to download from the source directly
if "runs:/" in source:
    parts = source.replace("runs:/", "").split("/")
    source_run_id = parts[0]
    artifact_path = "/".join(parts[1:])
    
    print(f"\nParsed source:")
    print(f"  Run ID: {source_run_id}")
    print(f"  Artifact path: {artifact_path}")
    
    # Try to download
    print(f"\nAttempting to download '{artifact_path}'...")
    try:
        local_path = mlflow.artifacts.download_artifacts(
            run_id=source_run_id,
            artifact_path=artifact_path
        )
        print(f"✓ Downloaded to: {local_path}")
        
        # Check if it's a file
        import pathlib
        p = pathlib.Path(local_path)
        if p.is_file():
            print(f"  Size: {p.stat().st_size} bytes")
        else:
            print(f"  ERROR: Not a file, it's a directory!")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        import traceback
        traceback.print_exc()

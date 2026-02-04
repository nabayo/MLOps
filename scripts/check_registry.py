
import mlflow
from mlflow.tracking import MlflowClient
import os
import json

def check_registry():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    print(f"Checking registry at {tracking_uri}...")
    
    try:
        models = client.search_registered_models()
        print(f"Found {len(models)} registered models:")
        for model in models:
            print(f"Name: {model.name}")
            for v in model.latest_versions:
                print(f"  - Version: {v.version}, Stage: {v.current_stage}, Status: {v.status}")
                
        if len(models) == 0:
            print("Registry is empty (via search).")
            
        # Specific check
        try:
            m = client.get_registered_model("YOLOv11-Finger-Counter")
            print(f"Direct Lookup: Found {m.name}")
            for v in m.latest_versions:
                 print(f"  Version: {v.version}, Stage: {v.current_stage}")
        except Exception as e:
            print(f"Direct Lookup (YOLOv11-Finger-Counter) Failed: {e}")
            
        # Also check for the config name
        try:
            m = client.get_registered_model("yolov11-finger-counting")
            print(f"Direct Lookup: Found {m.name}")
            for v in m.latest_versions:
                 print(f"  Version: {v.version}, Stage: {v.current_stage}")
        except Exception as e:
            print(f"Direct Lookup (yolov11-finger-counting) Failed: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_registry()

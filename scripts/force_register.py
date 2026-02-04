
import mlflow
from mlflow.tracking import MlflowClient
import os

def force_register():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    model_name = "YOLOv11-Finger-Counter"
    
    # 1. Ensure Registered Model exists
    try:
        client.create_registered_model(model_name)
        print(f"Created Registered Model: {model_name}")
    except Exception:
        print(f"Registered Model {model_name} already exists (likely).")
        
    # 2. Find a suitable run
    experiment = client.get_experiment_by_name("Model_Recovery")
    if not experiment:
        print("Experiment 'Model_Recovery' not found.")
        return
        
    runs = client.search_runs(experiment.experiment_id)
    if not runs:
        print("No runs found in Model_Recovery.")
        return
        
    run = runs[0]
    run_id = run.info.run_id
    print(f"Using Run ID: {run_id}")
    
    # 3. Create Model Version
    # The artifact was logged to "weights/model.pt" inside the run.
    # So source URI is "runs:/<run_id>/weights/model.pt"
    source = f"runs:/{run_id}/weights/model.pt"
    
    print(f"Attempting to register version from {source}...")
    
    try:
        mv = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id
        )
        print(f"SUCCESS! Created Version {mv.version} status: {mv.status}")
    except Exception as e:
        print(f"FAILED to create model version: {e}")

if __name__ == "__main__":
    force_register()


import mlflow
import requests
import os
import sys

def debug_connection():
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    print(f"Target URI: {uri}")
    
    # 1. Test raw connection
    try:
        r = requests.get(f"{uri}/health")
        print(f"Health Check: {r.status_code}")
        print(f"Content: {r.text[:100]}")
    except Exception as e:
        print(f"Health Check Failed: {e}")
        
    # 2. Test API Create Experiment
    try:
        r = requests.post(f"{uri}/api/2.0/mlflow/experiments/create", json={"name": "Debug_Exp"})
        print(f"Create Exp: {r.status_code}")
        print(f"Response: {r.text}")
    except Exception as e:
        print(f"Create Exp Failed: {e}")

    # 3. Test MLflow Client
    print("\nMLflow Client Check:")
    mlflow.set_tracking_uri(uri)
    try:
        exp = mlflow.get_experiment_by_name("Debug_Exp")
        if exp:
            print(f"Found Experiment: {exp.experiment_id}")
        else:
            print("Experiment not found via Client (likely 404 on list)")
            
        with mlflow.start_run(run_name="debug_run"):
            print("Run started successfully!")
            mlflow.log_param("test", "value")
            
    except Exception as e:
        print(f"Client Error: {e}")

if __name__ == "__main__":
    debug_connection()

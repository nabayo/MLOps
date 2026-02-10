import os

import mlflow
from mlflow.tracking import MlflowClient


def promote_to_production():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = "YOLOv11-Finger-Counter"
    version = 1

    print(f"Promoting {model_name} version {version} to Production...")

    try:
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )
        print("Success!")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Failed: {e}")

    # Check search with filter
    print("\nChecking Search with Filter...")
    try:
        res = client.search_registered_models(f"name='{model_name}'")
        print(f"Filter Search Found: {len(res)}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"Filter Search Failed: {e}")


if __name__ == "__main__":
    promote_to_production()

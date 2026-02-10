import os

import mlflow
from mlflow.tracking import MlflowClient


def register_all_models():
    """
    Scans all MLflow runs and tries to register models blindly from weights/best.pt
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    print(f"🔌 Connecting to MLflow at {tracking_uri}...")
    client = MlflowClient()

    try:
        experiments = client.search_experiments()
        print(f"✅ Found {len(experiments)} experiments.")
    except Exception as e:  # pylint: disable=broad-except
        print(f"❌ Failed to connect: {e}")
        return

    registered_count = 0

    for exp in experiments:
        print(f"\n📂 Experiment: {exp.name} (ID: {exp.experiment_id})")

        try:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
        except Exception as _e:
            continue

        for run in runs:
            if run.info.status != "FINISHED":
                continue

            run_id = run.info.run_id
            run_name = run.info.run_name
            run_id = run.info.run_id
            run_name = run.info.run_name
            # Infer model name from run name (contains experiment + date)
            model_name = (
                run_name
                if run_name
                else run.data.params.get("model_name", "YOLOv11-Finger-Counter")
            )

            # Clean up model name (replace spaces, etc if needed, though run_name is usually safe)
            # Ensure it doesn't contain invalid characters for MLflow registry
            model_name = model_name.replace(":", "-").replace(" ", "_")

            print(f"  🏃 Run: {run_name} ({run_id})")

            # Check if registered
            try:
                if client.search_model_versions(f"run_id='{run_id}'"):
                    print("    ⏭ Already registered.")
                    continue
            except Exception as _e:
                pass

            # Try to register best.pt or last.pt
            registered_version = None

            for pt_file in ["weights/best.pt", "weights/last.pt", "best.pt", "last.pt"]:
                print(f"    ✨ Attempting to register {pt_file} to '{model_name}'...")
                try:
                    model_uri = f"runs:/{run_id}/{pt_file}"

                    # 1. Register Model
                    result = mlflow.register_model(model_uri=model_uri, name=model_name)
                    print(f"    ✅ Registered version {result.version}")

                    # 2. Update Description (Optional)
                    try:
                        client.update_model_version(
                            name=model_name,
                            version=result.version,
                            description=f"Recovered from run {run_name} (Artifact: {pt_file})",
                        )
                    except Exception as desc_error:
                        print(
                            f"      ⚠ Warning: Could not update description: {desc_error}"
                        )

                    registered_count += 1
                    registered_version = result
                    break  # Success

                except Exception as _e:
                    # If 404/RestException, it means artifact missing or something
                    # print(f"      warn: {_e}")
                    pass

            if not registered_version:
                print("    ❌ Failed to register any model artifact.")

    print(f"\n🎉 Registered {registered_count} new models.")


if __name__ == "__main__":
    register_all_models()

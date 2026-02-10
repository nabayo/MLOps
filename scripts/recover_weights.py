import os
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

# Setup environment matches docker-compose
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin_password_change_me"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"


def download_all_weights():
    print("Connecting to MLflow...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    print("Searching for experiments...")
    experiments = client.search_experiments()

    total_downloaded = 0

    for exp in experiments:
        print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
        runs = client.search_runs(
            exp.experiment_id, order_by=["attribute.start_time DESC"]
        )

        if not runs:
            print("  No runs found.")
            continue

        for run in runs:
            run_id = run.info.run_id
            status = run.info.status
            print(f"  Run: {run_id} ({status})")

            # Create a specific directory for this run to avoid collisions
            # models/<experiments_name>/<run_id>/
            run_dir = Path("models") / exp.name / run_id

            # Create a specific directory for this run
            # models/<experiments_name>/<run_id>/
            run_dir = Path("models") / exp.name / run_id

            # Since list_artifacts is buggy/404s on directories in this environment,
            # we blindly try to download known weight files.

            common_weight_paths = [
                "weights/best.pt",
                "weights/last.pt",
                "best.pt",
                "last.pt",
                "weights/yolov11n.pt",  # Configurable names
                "yolov26s.pt",
            ]

            files_downloaded_for_run = 0

            for artifact_path in common_weight_paths:
                try:
                    # Construct destination: preserve filename, flatten structure inside run_dir?
                    # Let's keep it simple: run_dir / filename
                    _filename = Path(artifact_path).name

                    # Check if already exists to avoid re-downloading?
                    # No, overwrite to be safe.

                    # We download to a temp dir then move?
                    # create run_dir if we are about to download

                    # client.download_artifacts with a file path downloads just that file.
                    # It puts it into dst_path.

                    print(f"    Attempting fetch: {artifact_path} ...", end="")
                    _local_path = client.download_artifacts(
                        run_id, artifact_path, dst_path=str(run_dir)
                    )
                    print(" ✅ Success!")
                    files_downloaded_for_run += 1
                    total_downloaded += 1

                except Exception:
                    # Silent fail for missing files
                    print(" (not found)")
                    pass

            if files_downloaded_for_run == 0:
                # If we didn't get any weights, maybe clean up the empty dir?
                pass

    print("\n" + "=" * 50)
    print(f"Download complete. Total files: {total_downloaded}")
    print("Check the 'models/' directory.")


if __name__ == "__main__":
    download_all_weights()

import os
import sys
import datetime
import traceback
from pathlib import Path
import mlflow
from dotenv import load_dotenv
import tempfile

import random
from mlflow.tracking import MlflowClient

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv()


def list_models() -> dict[str, tuple[str, str, list[str]]]:
    """
    List all models in the MLflow tracking server.
    Returns a dictionary of models with their run IDs and their weights paths.
    """

    dict_weights: dict[str, tuple[str, str, list[str]]] = {}

    try:
        # Ensure we are connected to the right URI
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        print(f"MLFLOW_TRACKING_URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient()

        experiments = client.search_experiments()
        print(f"Found {len(experiments)} experiments:")

        for exp in experiments:
            print(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})")
            print("-" * 30)

            runs = client.search_runs(
                exp.experiment_id, order_by=["attribute.start_time DESC"]
            )
            if not runs:
                print("  No runs found.")
                continue

            for run in runs:
                run_id = run.info.run_id
                status = run.info.status
                start_time = run.info.start_time

                # Format time
                try:
                    dt = datetime.datetime.fromtimestamp(start_time / 1000)
                    time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                except Exception as _e:
                    time_str = str(start_time)

                print(f"  Run ID: {run_id}")
                print(f"    Date: {time_str}")
                print(f"    Status: {status}")
                if "model_architecture" in run.data.params:
                    print(f"    Arch: {run.data.params['model_architecture']}")

                # Check for weights
                weights_found = []

                # 1. Try listing root artifacts (usually works)
                try:
                    artifacts_root = client.list_artifacts(run_id)
                    weights_found.extend(
                        [a.path for a in artifacts_root if a.path.endswith(".pt")]
                    )
                except Exception:
                    # Root listing failed, not critical
                    pass

                # 2. Check for common weights using "Blind Download" trick
                # list_artifacts('weights') fails with 404, so we test existence by trying to download
                # to a temp directory.
                potential_weights = ["weights/best.pt", "weights/last.pt"]

                with tempfile.TemporaryDirectory() as temp_dir:
                    for weight_path in potential_weights:
                        try:
                            # We don't need the file, just want to know if it downloads without error
                            # This is SLOW (downloads full file) but reliable given the API issues
                            client.download_artifacts(
                                run_id, weight_path, dst_path=temp_dir
                            )
                            weights_found.append(weight_path)
                        except Exception:
                            # Failed to download, so it likely doesn't exist
                            pass

                if weights_found:
                    print(f"    Weights (verified in MLflow): {weights_found}")
                else:
                    print("    Weights: None found")

                print("")

                #
                if weights_found:
                    dict_weights[run_id] = (exp.name, status, weights_found)

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error listing experiments: {e}")
        traceback.print_exc()

    #
    return dict_weights


#


def load_dataset() -> list[Path]:
    """
    Load all valid images from the dataset directory.
    """
    dataset_dir = PROJECT_ROOT / "dataset"
    valid_exts = {".jpg", ".jpeg", ".png"}
    image_paths = []

    if not dataset_dir.exists():
        print(f"Warning: Dataset directory not found at {dataset_dir}")
        return []

    # Walk recursively
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in valid_exts:
                image_paths.append(file_path)

    return image_paths


def my_main():

    dict_weights = list_models()
    # print(dict_weights)

    image_paths = load_dataset()
    if not image_paths:
        print("No images found in dataset to test.")
        return

    client = MlflowClient()

    #
    for run_id, (exp_name, status, weights_found) in dict_weights.items():
        print("\n" + "*" * 60)
        print(f"Testing Run: {run_id} ({exp_name})")
        print(f"Available weights: {weights_found}")

        # Test each weight file found for this run
        for weight_file in weights_found:
            print(f"\n  -- Testing weights: {weight_file} --")

            # Download weights to a temp file
            # We must keep the temp file alive during inference
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    local_weight_path = client.download_artifacts(
                        run_id, weight_file, dst_path=temp_dir
                    )
                    print(f"    Downloaded to: {local_weight_path}")

                    try:
                        from ultralytics import YOLO

                        model = YOLO(local_weight_path)

                        # Select 5 random images
                        test_images = random.sample(
                            image_paths, min(5, len(image_paths))
                        )

                        for img_path in test_images:
                            print(f"    Inference on: {img_path.name}")
                            # Predict
                            results = model.predict(
                                source=str(img_path), save=False, verbose=False
                            )

                            for r in results:
                                num_obj = len(r.boxes)
                                print(f"      -> {num_obj} objects detected")
                                for box in r.boxes:
                                    conf = float(box.conf[0])
                                    cls_id = int(box.cls[0])
                                    cls_name = (
                                        model.names[cls_id]
                                        if hasattr(model, "names")
                                        else str(cls_id)
                                    )
                                    print(f"         - {cls_name}: {conf:.2f}")

                    except Exception as e:  # pylint: disable=broad-except
                        print(f"    ❌ Inference failed: {e}")
                        traceback.print_exc()

                except Exception as e:  # pylint: disable=broad-except
                    print(f"    ❌ Failed to download weights: {e}")

        print("*" * 60)


if __name__ == "__main__":
    my_main()

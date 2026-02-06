import os
import sys
import argparse
import datetime
import traceback
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import mlflow
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Load environment variables
load_dotenv()

def setup_args():
    parser = argparse.ArgumentParser(description="Test MLOps components: Preprocessing, Inference (Local/Registry)")
    parser.add_argument("--image", type=str, help="Path to an image to test")
    parser.add_argument("--weights", type=str, help="Path to local model weights (.pt)")
    parser.add_argument("--model-name", type=str, default="yolov11-finger-counting", help="Registered model name")
    parser.add_argument("--stage", type=str, default="Production", help="Model stage to load from registry")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Directory to save results")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable preprocessing test")
    parser.add_argument("--no-local", action="store_true", help="Disable local inference test")
    parser.add_argument("--no-registry", action="store_true", help="Disable registry inference test")
    return parser.parse_args()

def find_test_image(dataset_dir: Path) -> Optional[Path]:
    """Find a valid image in the dataset directory."""
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for path in dataset_dir.rglob('*'):
        if path.suffix.lower() in valid_exts:
            return path
    return None

def test_preprocessing(image_path: Path, output_dir: Path):
    print("\n" + "="*50)
    print("Testing Preprocessing")
    print("="*50)
    
    try:
        from src.preprocessing import FastFaceBlurStep
        # Test directly using the class if api import fails
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"cv2.imread failed for {image_path}, trying PIL...")
            try:
                from PIL import Image
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Could not read image with PIL either: {e}")
                return

        print(f"Original image shape: {img.shape}")
        
        # Apply blur directly
        print("Applying FastFaceBlurStep directly...")
        blur_step = FastFaceBlurStep(blur_kernel_size=51)
        processed_img = blur_step.process(img)
        
        print(f"Processed image shape: {processed_img.shape}")
        
        # Save output
        output_path = output_dir / f"preprocessed_{image_path.name}"
        cv2.imwrite(str(output_path), processed_img)
        print(f"Preprocessing result saved to: {output_path}")
        
    except ImportError:
         print("Could not import 'src.preprocessing'. Check path/dependencies.")
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        traceback.print_exc()

    print("\n" + "="*50)
    print("🔍 Searching for local model weights (*.pt)...")
    print("="*50)
    
    found_any = False
    # Walk typical directories
    search_paths = [Path("experiments"), Path("models"), Path("runs"), Path("mlflow")]
    # Also check current directory
    search_paths.append(Path("."))

    for search_root in search_paths:
        if not search_root.exists():
            continue
            
        print(f"Scanning {search_root}...")
        try:
             # Use concise walking for .pt files
            for root, dirs, files in os.walk(search_root):
                # Skip .git and typical huge dirs if recursing from .
                if ".git" in dirs:
                    dirs.remove(".git")
                if "dataset" in dirs: # Skip dataset, unlikely to have trained weights and is huge
                    dirs.remove("dataset")
                if ".venv" in dirs:
                    dirs.remove(".venv")
                    
                for file in files:
                    if file.endswith(".pt"):
                        full_path = Path(root) / file
                        print(f"  FOUND: {full_path}")
                        found_any = True
        except Exception as e:
             print(f"  Error scanning {search_root}: {e}")

    if not found_any:
        print("  ❌ No .pt files found in local directories.")
    else:
        print("  ✓ Local search complete.")


def test_local_inference(image_path: Path, weights_path: Path):
    print("\n" + "="*50)
    print("Testing Local Inference")
    print("="*50)
    
    if not weights_path.exists():
        print(f"Weights file not found: {weights_path}")
        return

    try:
        from ultralytics import YOLO
        print(f"Loading model from: {weights_path}")
        model = YOLO(weights_path)
        
        print(f"Running inference on: {image_path}")
        results = model.predict(source=str(image_path), save=False, verbose=False)
        
        for r in results:
            print(f"found {len(r.boxes)} objects")
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f" - {cls_name} ({conf:.2f})")
                
        print("Local inference successful")
        
    except ImportError:
         print("Could not import 'ultralytics'. Check installation.")
    except Exception as e:
        print(f"Local inference failed: {e}")

def test_registry_inference(image_path: Path, model_name: str, stage: str):
    print("\n" + "="*50)
    print("Testing Registry Inference")
    print("="*50)
    
    try:
        from serving.api import load_model_from_registry, current_model
        
        print(f"Attempting to load model '{model_name}' (Stage: {stage}) from MLflow...")
        load_model_from_registry(model_name, stage=stage)
        
        if current_model:
            print("Model loaded successfully from registry")
            
            print(f"Running inference on: {image_path}")
            model = current_model
            results = model.predict(source=str(image_path), save=False, verbose=False)
             
            for r in results:
                print(f"found {len(r.boxes)} objects")
                
            print("Registry inference successful")
        else:
            print("Failed to load model (current_model is None)")
            
    except Exception as e:
        print(f"Registry inference failed: {e}")
        print("Hint: Ensure MLflow is running and the model exists in the registry.")

def main():
    args = setup_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Find Image
    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # Try to find one in the default dataset location
        dataset_dir = PROJECT_ROOT / "dataset" / "Photos_Floutées" / "new_version"
        image_path = find_test_image(dataset_dir)
        if not image_path:
             # Fallback to just dataset/
            image_path = find_test_image(PROJECT_ROOT / "dataset")
    
    if not image_path or not image_path.exists():
        print("No test image found. Please specify one with --image")
        # Try to use a placeholder if absolute verified paths are missing, but for now just exit
        return

    print(f"Using test image: {image_path}")

    # 2. Test Preprocessing
    if not args.no_preprocess:
        test_preprocessing(image_path, output_dir)

    # 3. Test Local Inference
    if not args.no_local:
        if args.weights:
            test_local_inference(image_path, Path(args.weights))
        else:
            print("\nSkipping local inference test (no --weights provided)")
            print("  Use --weights /path/to/best.pt to test local model")

    # 4. Test Registry Inference
    if not args.no_registry:
        test_registry_inference(image_path, args.model_name, args.stage)


def my_main():
    print("\n" + "="*50)
    print("Listing Experiments and Trained Weights")
    print("="*50)

    try:
        from mlflow.tracking import MlflowClient
        import mlflow
        
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
            
            runs = client.search_runs(exp.experiment_id, order_by=["attribute.start_time DESC"])
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
                except:
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
                    weights_found.extend([a.path for a in artifacts_root if a.path.endswith('.pt')])
                except Exception:
                    # Root listing failed, not critical
                    pass

                # 2. Check for common weights using "Blind Download" trick
                # list_artifacts('weights') fails with 404, so we test existence by trying to download
                # to a temp directory.
                potential_weights = ["weights/best.pt", "weights/last.pt"]
                
                import tempfile
                import shutil
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    for weight_path in potential_weights:
                        try:
                            # We don't need the file, just want to know if it downloads without error
                            # This is SLOW (downloads full file) but reliable given the API issues
                            client.download_artifacts(run_id, weight_path, dst_path=temp_dir)
                            weights_found.append(weight_path)
                        except Exception:
                            # Failed to download, so it likely doesn't exist
                            pass

                if weights_found:
                    print(f"    Weights (verified in MLflow): {weights_found}")
                else:
                    print("    Weights: None found")
                
                print("")

    except Exception as e:
        print(f"Error listing experiments: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    my_main()

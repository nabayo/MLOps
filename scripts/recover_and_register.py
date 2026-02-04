
import boto3
import mlflow
import os
import tempfile
from ultralytics import YOLO

def recover_and_register():
    # MinIO config
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("MINIO_BUCKET_NAME", "mlflow-artifacts")
    
    # MLflow config
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("Model_Recovery")

    print(f"🔌 Connecting to MinIO at {endpoint}...")
    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret
    )
    
    # List all .pt files
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
    except Exception as e:
        print(f"❌ Failed to list bucket: {e}")
        return

    if 'Contents' not in response:
        print("Empty bucket.")
        return

    for obj in response['Contents']:
        key_path = obj['Key']
        if key_path.endswith('best.pt') or key_path.endswith('last.pt'):
            print(f"\n📦 Found checkpoint: {key_path}")
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = os.path.join(tmp_dir, "model.pt")
                print(f"  ⬇ Downloading...")
                s3.download_file(bucket_name, key_path, local_path)
                
                try:
                    # Parse run ID from key for naming
                    parts = key_path.split('/')
                    if len(parts) >= 2:
                        run_id_origin = parts[1]
                    else:
                        run_id_origin = "unknown"

                    print(f"  🔄 Loading model with YOLO...")
                    model = YOLO(local_path)
                    
                    model_name = "YOLOv11-Finger-Counter"
                    run_name = f"recovered_{run_id_origin}"
                    
                    print(f"  ⬆ Logging and registering to '{model_name}'...")
                    with mlflow.start_run(run_name=run_name) as run:
                        mlflow.log_param("original_s3_key", key_path)
                        mlflow.log_param("recovery_source", "minio_direct")
                        
                        # Log as artifact
                        mlflow.log_artifact(local_path, "weights")
                        print("    ✅ Artifact logged.")
                        
                        try:
                            # Register model using run artifact URI
                            model_uri = f"runs:/{run.info.run_id}/weights/model.pt"
                            print(f"    ✨ Registering {model_uri} to '{model_name}'...")
                            
                            mlflow.register_model(model_uri, model_name)
                            print("    ✅ Registration API called successfully.")
                        except Exception as reg_error:
                            print(f"    ❌ Registration failed: {reg_error}")
                        
                except Exception as e:
                    print(f"  ❌ Error loading/registering: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    recover_and_register()

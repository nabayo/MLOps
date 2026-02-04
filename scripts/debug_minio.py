
import boto3
import os

def list_minio_content():
    endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("MINIO_BUCKET_NAME", "mlflow-artifacts")

    print(f"Connecting to MinIO at {endpoint}...")
    print(f"Bucket: {bucket_name}")
    print(f"User: {key}")

    s3 = boto3.client(
        's3',
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret
    )

    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            print(f"Found {len(response['Contents'])} objects:")
            for obj in response['Contents']:
                if obj['Key'].endswith('.pt'):
                    print(f" - {obj['Key']} (Size: {obj['Size']})")
        else:
            print("Bucket is empty.")
    except Exception as e:
        print(f"Error listing bucket: {e}")

if __name__ == "__main__":
    list_minio_content()

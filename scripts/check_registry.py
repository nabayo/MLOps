
import mlflow
from mlflow.tracking import MlflowClient
import os
import json

def check_registry():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    output_path = "/app/experiments/registry_dump.txt"
    with open(output_path, "w") as f:
        f.write(f"Registry Check at {tracking_uri}\n")
        f.write("=" * 50 + "\n")
        
        try:
            models = client.search_registered_models()
            f.write(f"Found {len(models)} registered models:\n")
            for model in models:
                f.write(f"\nModel: {model.name}\n")
                f.write(f"  Description: {model.description}\n")
                if model.latest_versions:
                    for v in model.latest_versions:
                        f.write(f"  - Version: {v.version}\n")
                        f.write(f"    Stage: {v.current_stage}\n")
                        f.write(f"    Status: {v.status}\n")
                        f.write(f"    Source: {v.source}\n")
                        f.write(f"    Run ID: {v.run_id}\n")
                else:
                    f.write("  (No versions found)\n")
                    
            if len(models) == 0:
                f.write("\nRegistry is empty (via search).\n")
                
            # DEBUG: Check artifacts for the first available run to see why registration might be failing
            try:
                f.write("\n=== Artifact Inspection ===\n")
                experiments = client.search_experiments()
                if experiments:
                    runs = client.search_runs(experiments[0].experiment_id)
                    if runs:
                        run = runs[0]
                        f.write(f"Inspecting Run: {run.info.run_id} ({run.info.run_name})\n")
                        artifacts = client.list_artifacts(run.info.run_id)
                        f.write("Root Artifacts:\n")
                        for art in artifacts:
                            f.write(f"  - {art.path} (is_dir={art.is_dir})\n")
                            if art.is_dir:
                                sub = client.list_artifacts(run.info.run_id, art.path)
                                for s in sub:
                                    f.write(f"    - {s.path}\n")
                    else:
                        f.write("No runs found to inspect.\n")
                else:
                    f.write("No experiments found.\n")
            except Exception as e:
                f.write(f"Error inspecting artifacts: {e}\n")

        except Exception as e:
            f.write(f"\nError searching models: {e}\n")
            
    print(f"Registry dump written to {output_path}")

if __name__ == "__main__":
    check_registry()

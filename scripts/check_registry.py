"""
Check MLflow registry for registered models.
"""

import os
import mlflow

from mlflow.tracking import MlflowClient


def check_registry() -> None:
    """
    Check MLflow registry for registered models.
    """

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    output_path = "/app/experiments/registry_dump.txt"
    with open(output_path, "w", encoding="utf-8") as f:
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

            # DEBUG: Check artifacts for the all run
            #        to see why registration might be failing
            try:
                f.write("\n=== Artifact Inspection (All Runs) ===\n")
                experiments = client.search_experiments()
                for exp in experiments:
                    f.write(f"\nExperiment: {exp.name} (ID: {exp.experiment_id})\n")
                    runs = client.search_runs(exp.experiment_id)
                    if runs:
                        for run in runs[:5]:  # Inspect top 5 runs
                            f.write(f"  Run: {run.info.run_id} ({run.info.run_name})\n")
                            f.write(f"    Artifact URI: {run.info.artifact_uri}\n")
                            try:
                                artifacts = client.list_artifacts(run.info.run_id)
                                if artifacts:
                                    f.write("    Artifacts:\n")
                                    for art in artifacts:
                                        f.write(f"      - {art.path}\n")
                                        if art.path == "weights" and art.is_dir:
                                            sub = client.list_artifacts(
                                                run.info.run_id, "weights"
                                            )
                                            for s in sub:
                                                f.write(f"        - {s.path}\n")
                                else:
                                    f.write("    (No artifacts found)\n")
                            except Exception as e:  # pylint: disable=broad-except
                                f.write(f"    Error listing artifacts: {e}\n")
                    else:
                        f.write("  No runs found.\n")
            except Exception as e:  # pylint: disable=broad-except
                f.write(f"Error inspecting artifacts: {e}\n")

        except Exception as e:  # pylint: disable=broad-except
            f.write(f"\nError searching models: {e}\n")

    print(f"Registry dump written to {output_path}")


if __name__ == "__main__":
    check_registry()

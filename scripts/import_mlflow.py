#!/usr/bin/env python3
"""
MLflow Import Script - Restore experiments, runs, and artifacts from a backup zip.

This script imports MLflow data with collision handling:
- Skips existing experiments (by name)
- Skips existing runs (by run_id)
- Skips existing model versions
- Skips existing artifact files
"""

from typing import Dict, Any, Set

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus


def import_mlflow_data(
    zip_path: str, skip_existing: bool = True, dry_run: bool = False
) -> Dict[str, Any]:
    """
    Import MLflow data from a backup zip file.

    Args:
        zip_path: Path to backup zip file
        skip_existing: If True, skip existing data (default collision handling)
        dry_run: If True, only show what would be imported without making changes

    Returns:
        Dictionary with import statistics
    """
    zip_file = Path(zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(f"Backup file not found: {zip_path}")

    temp_dir = Path(f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    try:
        print("=" * 70)
        print("🔄 MLflow Data Import")
        if dry_run:
            print("🔍 DRY RUN MODE  - No changes will be made")
        print("=" * 70)

        # Extract zip
        print(f"\n📦 Extracting backup: {zip_file.name}")
        with zipfile.ZipFile(zip_file, "r") as zipf:
            zipf.extractall(temp_dir)

        # Load metadata
        with open(temp_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        print(f"📅 Backup date: {metadata['export_date']}")
        print(
            f"📊 Contains: {metadata['total_experiments']} experiments, "
            f"{metadata['total_runs']} runs, {metadata['total_models']} models"
        )

        if dry_run:
            print("\n🔍 Analyzing backup contents...")

        client = MlflowClient()
        stats = {
            "experiments_created": 0,
            "experiments_skipped": 0,
            "runs_created": 0,
            "runs_skipped": 0,
            "models_created": 0,
            "model_versions_created": 0,
            "model_versions_skipped": 0,
            "artifacts_uploaded": 0,
            "artifacts_skipped": 0,
        }

        # Get existing experiments and runs
        existing_experiments = {
            exp.name: exp for exp in client.search_experiments(view_type=3)
        }  # ALL
        existing_runs: Set[str] = set()
        for exp in existing_experiments.values():
            runs = client.search_runs([exp.experiment_id])
            existing_runs.update(run.info.run_id for run in runs)

        # Import experiments
        print("\n📦 Importing experiments...")
        experiments_file = temp_dir / "experiments" / "experiments.json"
        with open(experiments_file, "r") as f:
            experiments_data = json.load(f)

        experiment_id_mapping = {}  # old_id -> new_id

        for exp_data in experiments_data:
            exp_name = exp_data["name"]

            if exp_name in existing_experiments:
                if skip_existing:
                    stats["experiments_skipped"] += 1
                    # Map to existing experiment
                    experiment_id_mapping[exp_data["experiment_id"]] = (
                        existing_experiments[exp_name].experiment_id
                    )
                    print(f"  ⊘ Skipped (exists): {exp_name}")
                    continue

            if not dry_run:
                new_exp_id = client.create_experiment(
                    exp_name, tags=exp_data.get("tags", {})
                )
                experiment_id_mapping[exp_data["experiment_id"]] = new_exp_id
                print(f"  ✓ Created: {exp_name} (ID: {new_exp_id})")
            else:
                print(f"  ✓ Would create: {exp_name}")

            stats["experiments_created"] += 1

        # Import runs
        print("\n🏃 Importing runs...")
        runs_dir = temp_dir / "runs"

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue

            with open(run_file, "r") as f:
                run_data = json.load(f)

            run_id = run_data["info"]["run_id"]
            run_name = run_data["info"].get("run_name", run_id[:8])

            if run_id in existing_runs:
                if skip_existing:
                    stats["runs_skipped"] += 1
                    print(f"  ⊘ Skipped (exists): {run_name}")
                    continue

            if not dry_run:
                # Get mapped experiment ID
                old_exp_id = run_data["info"]["experiment_id"]
                new_exp_id = experiment_id_mapping.get(old_exp_id)

                if new_exp_id is None:
                    print(
                        f"  ⚠ Warning: Experiment not found for run {run_name}, skipping"
                    )
                    stats["runs_skipped"] += 1
                    continue

                # Create run
                run = client.create_run(
                    experiment_id=new_exp_id,
                    run_name=run_name,
                    tags=run_data["data"].get("tags", {}),
                )

                # Log parameters
                for key, value in run_data["data"].get("params", {}).items():
                    client.log_param(run.info.run_id, key, value)

                # Log metrics
                for key, value in run_data["data"].get("metrics", {}).items():
                    client.log_metric(run.info.run_id, key, value)

                # Upload artifacts
                artifacts_dir = run_dir / "artifacts"
                if artifacts_dir.exists():
                    for root, dirs, files in os.walk(artifacts_dir):
                        for file in files:
                            file_path = Path(root) / file
                            rel_path = file_path.relative_to(artifacts_dir)

                            # Check if artifact directory exists
                            artifact_path = (
                                str(rel_path.parent)
                                if rel_path.parent != Path(".")
                                else ""
                            )

                            try:
                                client.log_artifact(
                                    run.info.run_id, str(file_path), artifact_path
                                )
                                stats["artifacts_uploaded"] += 1
                            except Exception as e:
                                print(
                                    f"    ⚠ Warning: Could not upload {rel_path}: {e}"
                                )
                                stats["artifacts_skipped"] += 1

                # End run with appropriate status
                end_status = RunStatus.to_string(RunStatus.FINISHED)
                client.set_terminated(run.info.run_id, end_status)

                print(
                    f"  ✓ Created: {run_name} ({stats['artifacts_uploaded']} artifacts)"
                )
            else:
                # Count artifacts in dry run
                artifacts_dir = run_dir / "artifacts"
                if artifacts_dir.exists():
                    artifact_count = sum(
                        1 for _ in artifacts_dir.rglob("*") if _.is_file()
                    )
                    print(f"  ✓ Would create: {run_name} ({artifact_count} artifacts)")

            stats["runs_created"] += 1

        # Import registered models
        print("\n🎯 Importing registered models...")
        models_file = temp_dir / "models" / "models.json"

        if models_file.exists():
            with open(models_file, "r") as f:
                models_data = json.load(f)

            # Get existing models
            existing_models = {
                model.name: model for model in client.search_registered_models()
            }

            for model_data in models_data:
                model_name = model_data["name"]

                # Create or get model
                if model_name in existing_models:
                    if skip_existing:
                        print(f"  ⊘ Model exists: {model_name}")
                        _model = existing_models[model_name]
                    else:
                        if dry_run:
                            print(f"  ✓ Model exists: {model_name}")
                        _model = existing_models[model_name]
                else:
                    if not dry_run:
                        client.create_registered_model(
                            model_name,
                            tags=model_data.get("tags", {}),
                            description=model_data.get("description"),
                        )
                        print(f"  ✓ Created model: {model_name}")
                    else:
                        print(f"  ✓ Would create model: {model_name}")
                    stats["models_created"] += 1

                # Import versions
                existing_versions = set()
                if model_name in existing_models and not dry_run:
                    versions = client.search_model_versions(f"name='{model_name}'")
                    existing_versions = {v.version for v in versions}

                for version_data in model_data.get("versions", []):
                    version = version_data["version"]
                    run_id = version_data.get("run_id")

                    if version in existing_versions:
                        if skip_existing:
                            stats["model_versions_skipped"] += 1
                            print(f"    ⊘ Version {version} exists")
                            continue

                    if not dry_run and run_id:
                        try:
                            # Note: This requires the run to exist
                            mv = client.create_model_version(
                                model_name,
                                f"runs:/{run_id}/model",
                                run_id=run_id,
                                tags=version_data.get("tags", {}),
                                description=version_data.get("description"),
                            )

                            # Set stage if not None
                            current_stage = version_data.get("current_stage")
                            if current_stage and current_stage != "None":
                                client.transition_model_version_stage(
                                    model_name, mv.version, current_stage
                                )

                            print(f"    ✓ Created version {mv.version}")
                            stats["model_versions_created"] += 1
                        except Exception as e:
                            print(
                                f"    ⚠ Warning: Could not create version {version}: {e}"
                            )
                            stats["model_versions_skipped"] += 1
                    else:
                        print(f"    ✓ Would create version {version}")
                        stats["model_versions_created"] += 1

        # Print summary
        print("\n" + "=" * 70)
        if dry_run:
            print("🔍 Dry Run Summary (No changes made)")
        else:
            print("✅ Import Complete!")
        print("=" * 70)
        print(
            f"📦 Experiments: {stats['experiments_created']} created, {stats['experiments_skipped']} skipped"
        )
        print(
            f"🏃 Runs: {stats['runs_created']} created, {stats['runs_skipped']} skipped"
        )
        print(f"🎯 Models: {stats['models_created']} created")
        print(
            f"📦 Model Versions: {stats['model_versions_created']} created, {stats['model_versions_skipped']} skipped"
        )
        print(
            f"📁 Artifacts: {stats['artifacts_uploaded']} uploaded, {stats['artifacts_skipped']} skipped"
        )

        return stats

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import MLflow data from a backup zip file"
    )
    parser.add_argument("backup_file", help="Path to backup zip file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data (default: skip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without making changes",
    )
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: from env)"
    )

    args = parser.parse_args()

    # Set tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    try:
        stats = import_mlflow_data(
            args.backup_file, skip_existing=not args.overwrite, dry_run=args.dry_run
        )
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Import failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
MLflow Export Script - Backup experiments, runs, and artifacts to a zip file.

This script exports:
- All experiments and their metadata
- All runs with parameters, metrics, and tags
- Model artifacts and files
- Run artifacts
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import mlflow
from mlflow.tracking import MlflowClient


def export_mlflow_data(output_dir: str = "backups", backup_name: str = None) -> str:
    """
    Export all MLflow data to a zip file.

    Args:
        output_dir: Directory to save the backup zip
        backup_name: Custom backup name (default: mlflow_backup_YYYYMMDD_HHMMSS.zip)

    Returns:
        Path to created zip file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate backup name
    if backup_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"mlflow_backup_{timestamp}.zip"

    if not backup_name.endswith('.zip'):
        backup_name += '.zip'

    zip_path = output_path / backup_name
    temp_dir = output_path / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 70)
        print("🔄 MLflow Data Export")
        print("=" * 70)

        client = MlflowClient()

        # Create backup structure
        experiments_dir = temp_dir / "experiments"
        runs_dir = temp_dir / "runs"
        models_dir = temp_dir / "models"
        experiments_dir.mkdir()
        runs_dir.mkdir()
        models_dir.mkdir()

        # Export experiments
        print("\n📦 Exporting experiments...")
        experiments = client.search_experiments()
        experiments_data = []

        for exp in experiments:
            exp_data = {
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'tags': exp.tags
            }
            experiments_data.append(exp_data)
            print(f"  ✓ Experiment: {exp.name} (ID: {exp.experiment_id})")

        with open(experiments_dir / "experiments.json", 'w') as f:
            json.dump(experiments_data, f, indent=2)

        # Export runs
        print("\n🏃 Exporting runs...")
        total_runs = 0

        for exp in experiments:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])

            for run in runs:
                total_runs += 1
                run_dir = runs_dir / run.info.run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                # Export run metadata
                run_data = {
                    'info': {
                        'run_id': run.info.run_id,
                        'experiment_id': run.info.experiment_id,
                        'run_name': run.info.run_name,
                        'status': run.info.status,
                        'start_time': run.info.start_time,
                        'end_time': run.info.end_time,
                        'artifact_uri': run.info.artifact_uri,
                    },
                    'data': {
                        'params': run.data.params,
                        'metrics': run.data.metrics,
                        'tags': run.data.tags
                    }
                }

                with open(run_dir / "run.json", 'w') as f:
                    json.dump(run_data, f, indent=2)

                # Export artifacts
                try:
                    artifacts_dir = run_dir / "artifacts"
                    artifacts_dir.mkdir(exist_ok=True)

                    # Download all artifacts for this run
                    local_artifact_path = client.download_artifacts(run.info.run_id, "", dst_path=str(artifacts_dir))

                except Exception as e:
                    print(f"    ⚠ Warning: Could not download artifacts for run {run.info.run_id}: {e}")

                print(f"  ✓ Run: {run.info.run_name or run.info.run_id[:8]}")

        print(f"\n  Total runs exported: {total_runs}")

        # Export registered models
        print("\n🎯 Exporting registered models...")
        try:
            models = client.search_registered_models()
            models_data = []

            for model in models:
                model_versions = client.search_model_versions(f"name='{model.name}'")

                model_data = {
                    'name': model.name,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'description': model.description,
                    'tags': model.tags,
                    'versions': []
                }

                for version in model_versions:
                    version_data = {
                        'version': version.version,
                        'creation_timestamp': version.creation_timestamp,
                        'last_updated_timestamp': version.last_updated_timestamp,
                        'description': version.description,
                        'run_id': version.run_id,
                        'status': version.status,
                        'current_stage': version.current_stage,
                        'tags': version.tags
                    }
                    model_data['versions'].append(version_data)

                models_data.append(model_data)
                print(f"  ✓ Model: {model.name} ({len(model_versions)} versions)")

            with open(models_dir / "models.json", 'w') as f:
                json.dump(models_data, f, indent=2)

        except Exception as e:
            print(f"  ⚠ Warning: Could not export models: {e}")

        # Create metadata file
        metadata = {
            'export_date': datetime.now().isoformat(),
            'mlflow_tracking_uri': mlflow.get_tracking_uri(),
            'total_experiments': len(experiments_data),
            'total_runs': total_runs,
            'total_models': len(models_data) if 'models_data' in locals() else 0
        }

        with open(temp_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create zip file
        print(f"\n📦 Creating backup archive: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)

        file_size_mb = zip_path.stat().st_size / (1024 * 1024)

        print("\n" + "=" * 70)
        print("✅ Export Complete!")
        print("=" * 70)
        print(f"📁 Backup file: {zip_path}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"📦 Experiments: {metadata['total_experiments']}")
        print(f"🏃 Runs: {metadata['total_runs']}")
        print(f"🎯 Models: {metadata['total_models']}")

        return str(zip_path)

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export MLflow data to a backup zip file")
    parser.add_argument("--output-dir", default="backups", help="Output directory for backup")
    parser.add_argument("--name", help="Custom backup name (default: auto-generated)")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (default: from env)")

    args = parser.parse_args()

    # Set tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    try:
        backup_path = export_mlflow_data(args.output_dir, args.name)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

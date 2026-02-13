#!/usr/bin/env python3
"""
MLflow Export Script - Backup experiments, runs, and artifacts to a zip file.

This script exports:
- All experiments and their metadata
- All runs with parameters, metrics, and tags
- Model artifacts and files
- Run artifacts
"""

from typing import Optional, Any

import os
import sys
import json
import zipfile
import shutil
import traceback
import argparse

from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient


class MLflowExporter:
    """
    Handles the export of MLflow data to a zip archive.
    """

    def __init__(self, output_dir: str, backup_name: Optional[str] = None):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Generate backup name
        if backup_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_name = f"mlflow_backup_{timestamp}.zip"
        else:
            self.backup_name = backup_name

        if not self.backup_name.endswith(".zip"):
            self.backup_name += ".zip"

        self.zip_path = self.output_path / self.backup_name
        self.temp_dir = (
            self.output_path / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.client = MlflowClient()

        # Directories
        self.experiments_dir = self.temp_dir / "experiments"
        self.runs_dir = self.temp_dir / "runs"
        self.models_dir = self.temp_dir / "models"

        self.experiments_dir.mkdir()
        self.runs_dir.mkdir()
        self.models_dir.mkdir()

        # Stats
        self.data_experiments: list[dict[str, Any]] = []
        self.data_models: list[dict[str, Any]] = []
        self.total_runs = 0

    def export(self) -> str:
        """
        Execute the export process.
        """
        try:
            print("=" * 70)
            print("🔄 MLflow Data Export")
            print("=" * 70)

            self._export_experiments()
            self._export_runs()
            self._export_models()
            self._write_metadata()
            self._create_zip()

            file_size_mb = self.zip_path.stat().st_size / (1024 * 1024)

            print("\n" + "=" * 70)
            print("✅ Export Complete!")
            print("=" * 70)
            print(f"📁 Backup file: {self.zip_path}")
            print(f"📊 Size: {file_size_mb:.2f} MB")
            print(f"📦 Experiments: {len(self.data_experiments)}")
            print(f"🏃 Runs: {self.total_runs}")
            print(f"🎯 Models: {len(self.data_models)}")

            return str(self.zip_path)

        finally:
            cleanup(self)

    def _export_experiments(self) -> None:
        """Export experiments metadata."""
        print("\n📦 Exporting experiments...")
        experiments = self.client.search_experiments()
        self.data_experiments = []

        for exp in experiments:
            exp_data = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags,
            }
            self.data_experiments.append(exp_data)
            print(f"  ✓ Experiment: {exp.name} (ID: {exp.experiment_id})")

        with open(
            self.experiments_dir / "experiments.json", "w", encoding="utf-8"
        ) as f:
            json.dump(self.data_experiments, f, indent=2)

    def _export_runs(self) -> None:
        """Export runs for all experiments."""
        print("\n🏃 Exporting runs...")
        self.total_runs = 0

        # Create explicit list of IDs to avoid issues if experiments list changes
        experiment_ids = [exp["experiment_id"] for exp in self.data_experiments]

        for exp_id in experiment_ids:
            runs = self.client.search_runs(experiment_ids=[exp_id])

            for run in runs:
                self.total_runs += 1
                self._export_single_run(run)
                print(f"  ✓ Run: {run.info.run_name or run.info.run_id[:8]}")

        print(f"\n  Total runs exported: {self.total_runs}")

    def _export_single_run(self, run: Any) -> None:
        """Export a single run's metadata and artifacts."""
        run_dir = self.runs_dir / run.info.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Export run metadata
        run_data = {
            "info": {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
            },
            "data": {
                "params": run.data.params,
                "metrics": run.data.metrics,
                "tags": run.data.tags,
            },
        }

        with open(run_dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2)

        # Export artifacts
        try:
            artifacts_dir = run_dir / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            # Download all artifacts for this run
            self.client.download_artifacts(
                run.info.run_id, "", dst_path=str(artifacts_dir)
            )

        except Exception as e:  # pylint: disable=broad-except
            print(
                f"    ⚠ Warning: Could not download artifacts for run {run.info.run_id}: {e}"
            )

    def _export_models(self) -> None:
        """Export registered models and their versions."""
        print("\n🎯 Exporting registered models...")
        try:
            models = self.client.search_registered_models()
            self.data_models = []

            for model in models:
                model_versions = self.client.search_model_versions(
                    f"name='{model.name}'"
                )

                model_data = {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "tags": model.tags,
                    "versions": [],
                }

                for version in model_versions:
                    version_data = {
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "description": version.description,
                        "run_id": version.run_id,
                        "status": version.status,
                        "current_stage": version.current_stage,
                        "tags": version.tags,
                    }
                    model_data["versions"].append(version_data)

                self.data_models.append(model_data)
                print(f"  ✓ Model: {model.name} ({len(model_versions)} versions)")

            with open(self.models_dir / "models.json", "w", encoding="utf-8") as f:
                json.dump(self.data_models, f, indent=2)

        except Exception as e:  # pylint: disable=broad-except
            print(f"  ⚠ Warning: Could not export models: {e}")

    def _write_metadata(self) -> None:
        """Write export metadata."""
        metadata = {
            "export_date": datetime.now().isoformat(),
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "total_experiments": len(self.data_experiments),
            "total_runs": self.total_runs,
            "total_models": len(self.data_models),
        }

        with open(self.temp_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _create_zip(self) -> None:
        """Compress the temporary directory into a zip file."""
        print(f"\n📦 Creating backup archive: {self.zip_path.name}")
        with zipfile.ZipFile(self.zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, _dirs, files in os.walk(self.temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.temp_dir)
                    zipf.write(file_path, arcname)


def cleanup(mlflow_exported: MLflowExporter) -> None:
    """Cleanup the exported MLflow data."""
    if mlflow_exported.temp_dir.exists():
        shutil.rmtree(mlflow_exported.temp_dir)


def main() -> None:
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Export MLflow data to a backup zip file"
    )
    parser.add_argument(
        "--output-dir", default="backups", help="Output directory for backup"
    )
    parser.add_argument("--name", help="Custom backup name (default: auto-generated)")
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: from env)"
    )

    args = parser.parse_args()

    # Set tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)

    try:
        exporter = MLflowExporter(args.output_dir, args.name)
        exporter.export()
        sys.exit(0)
    except Exception as e:  # pylint: disable=broad-except
        print(f"\n❌ Export failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

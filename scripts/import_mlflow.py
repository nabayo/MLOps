#!/usr/bin/env python3
"""
MLflow Import Script - Restore experiments, runs, and artifacts from a backup zip.

This script imports MLflow data with collision handling:
- Skips existing experiments (by name)
- Skips existing runs (by run_id)
- Skips existing model versions
- Skips existing artifact files
"""

from typing import Any, Dict, Set

import os
import sys
import json
import zipfile
import shutil
import traceback

from pathlib import Path
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus


class MLflowImporter:
    """
    Handles the import of MLflow data from a zip archive.
    """

    def __init__(
        self, zip_path: str, skip_existing: bool = True, dry_run: bool = False
    ):
        self.zip_file = Path(zip_path)
        if not self.zip_file.exists():
            raise FileNotFoundError(f"Backup file not found: {zip_path}")

        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.temp_dir = Path(f"temp_import_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.client = MlflowClient()

        # Stats
        self.stats = {
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

        # State
        self.experiment_id_mapping: Dict[str, str] = {}  # old_id -> new_id
        self.existing_runs: Set[str] = set()
        self.existing_experiments: Dict[str, Any] = {}

    def import_data(self) -> Dict[str, Any]:
        """
        Execute the import process.
        """
        try:
            self._print_header()
            self._extract_zip()
            self._load_metadata()

            if self.dry_run:
                print("\n🔍 Analyzing backup contents...")

            self._load_existing_state()
            self._import_experiments()
            self._import_runs()
            self._import_models()
            self._print_summary()

            return self.stats

        finally:
            self._cleanup()

    def _print_header(self) -> None:
        print("=" * 70)
        print("🔄 MLflow Data Import")
        if self.dry_run:
            print("🔍 DRY RUN MODE  - No changes will be made")
        print("=" * 70)

    def _extract_zip(self) -> None:
        print(f"\n📦 Extracting backup: {self.zip_file.name}")
        with zipfile.ZipFile(self.zip_file, "r") as zipf:
            zipf.extractall(self.temp_dir)

    def _load_metadata(self) -> None:
        with open(self.temp_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print(f"📅 Backup date: {metadata['export_date']}")
        print(
            f"📊 Contains: {metadata['total_experiments']} experiments, "
            f"{metadata['total_runs']} runs, {metadata['total_models']} models"
        )

    def _load_existing_state(self) -> None:
        """Pre-load existing experiments and runs to check for collisions."""
        self.existing_experiments = {
            exp.name: exp for exp in self.client.search_experiments(view_type=3)
        }  # ALL
        for exp in self.existing_experiments.values():
            runs = self.client.search_runs([exp.experiment_id])
            self.existing_runs.update(run.info.run_id for run in runs)

    def _import_experiments(self) -> None:
        """Import experiments from the backup."""
        print("\n📦 Importing experiments...")
        experiments_file = self.temp_dir / "experiments" / "experiments.json"
        with open(experiments_file, "r", encoding="utf-8") as f:
            experiments_data = json.load(f)

        for exp_data in experiments_data:
            self._import_single_experiment(exp_data)

    def _import_single_experiment(self, exp_data: Dict[str, Any]) -> None:
        exp_name = exp_data["name"]

        if exp_name in self.existing_experiments:
            if self.skip_existing:
                self.stats["experiments_skipped"] += 1
                # Map to existing experiment
                self.experiment_id_mapping[exp_data["experiment_id"]] = (
                    self.existing_experiments[exp_name].experiment_id
                )
                print(f"  ⊘ Skipped (exists): {exp_name}")
                return

        if not self.dry_run:
            new_exp_id = self.client.create_experiment(
                exp_name, tags=exp_data.get("tags", {})
            )
            self.experiment_id_mapping[exp_data["experiment_id"]] = new_exp_id
            print(f"  ✓ Created: {exp_name} (ID: {new_exp_id})")
        else:
            print(f"  ✓ Would create: {exp_name}")

        self.stats["experiments_created"] += 1

    def _import_runs(self) -> None:
        """Import runs from the backup."""
        print("\n🏃 Importing runs...")
        runs_dir = self.temp_dir / "runs"

        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            run_file = run_dir / "run.json"
            if not run_file.exists():
                continue

            with open(run_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)

            self._import_single_run(run_data, run_dir)

    def _import_single_run(self, run_data: Dict[str, Any], run_dir: Path) -> None:
        run_id = run_data["info"]["run_id"]
        run_name = run_data["info"].get("run_name", run_id[:8])

        if run_id in self.existing_runs and self.skip_existing:
            self.stats["runs_skipped"] += 1
            print(f"  ⊘ Skipped (exists): {run_name}")
            return

        if self.dry_run:
            self._dry_run_import_run(run_name, run_dir)
            self.stats["runs_created"] += 1
            return

        # Start actual import
        old_exp_id = run_data["info"]["experiment_id"]
        new_exp_id = self.experiment_id_mapping.get(old_exp_id)

        if new_exp_id is None:
            print(f"  ⚠ Warning: Experiment not found for run {run_name}, skipping")
            self.stats["runs_skipped"] += 1
            return

        # Create run
        run = self.client.create_run(
            experiment_id=new_exp_id,
            run_name=run_name,
            tags=run_data["data"].get("tags", {}),
        )

        # Log parameters and metrics
        self._log_run_data(run.info.run_id, run_data)

        # Upload artifacts
        self._upload_artifacts(run.info.run_id, run_dir)

        # End run
        end_status = RunStatus.to_string(RunStatus.FINISHED)
        self.client.set_terminated(run.info.run_id, end_status)

        print(
            f"  ✓ Created: {run_name} ({self.stats.get('artifacts_uploaded_last_run', 0)} artifacts)"
        )
        self.stats["runs_created"] += 1

    def _dry_run_import_run(self, run_name: str, run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        artifact_count = 0
        if artifacts_dir.exists():
            artifact_count = sum(1 for _ in artifacts_dir.rglob("*") if _.is_file())
        print(f"  ✓ Would create: {run_name} ({artifact_count} artifacts)")

    def _log_run_data(self, run_id: str, run_data: Dict[str, Any]) -> None:
        for key, value in run_data["data"].get("params", {}).items():
            self.client.log_param(run_id, key, value)

        for key, value in run_data["data"].get("metrics", {}).items():
            self.client.log_metric(run_id, key, value)

    def _upload_artifacts(self, run_id: str, run_dir: Path) -> None:
        artifacts_dir = run_dir / "artifacts"
        uploaded_count = 0
        if artifacts_dir.exists():
            for root, _dirs, files in os.walk(artifacts_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(artifacts_dir)
                    artifact_path = (
                        str(rel_path.parent) if rel_path.parent != Path(".") else ""
                    )

                    try:
                        self.client.log_artifact(run_id, str(file_path), artifact_path)
                        self.stats["artifacts_uploaded"] += 1
                        uploaded_count += 1
                    except Exception as e:  # pylint: disable=broad-except
                        print(f"    ⚠ Warning: Could not upload {rel_path}: {e}")
                        self.stats["artifacts_skipped"] += 1
        self.stats["artifacts_uploaded_last_run"] = uploaded_count

    def _import_models(self) -> None:
        """Import registered models."""
        print("\n🎯 Importing registered models...")
        models_file = self.temp_dir / "models" / "models.json"

        if not models_file.exists():
            return

        with open(models_file, "r", encoding="utf-8") as f:
            models_data = json.load(f)

        existing_models = {
            model.name: model for model in self.client.search_registered_models()
        }

        for model_data in models_data:
            self._import_single_model(model_data, existing_models)

    def _import_single_model(
        self, model_data: Dict[str, Any], existing_models: Dict[str, Any]
    ) -> None:
        model_name = model_data["name"]

        # Create or get model
        if model_name in existing_models:
            if self.skip_existing:
                print(f"  ⊘ Model exists: {model_name}")
            else:
                if self.dry_run:
                    print(f"  ✓ Model exists: {model_name}")
        else:
            if not self.dry_run:
                self.client.create_registered_model(
                    model_name,
                    tags=model_data.get("tags", {}),
                    description=model_data.get("description"),
                )
                print(f"  ✓ Created model: {model_name}")
            else:
                print(f"  ✓ Would create model: {model_name}")
            self.stats["models_created"] += 1

        self._import_model_versions(model_name, model_data, existing_models)

    def _import_model_versions(
        self,
        model_name: str,
        model_data: Dict[str, Any],
        existing_models: Dict[str, Any],
    ) -> None:
        existing_versions = set()
        if model_name in existing_models and not self.dry_run:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            existing_versions = {v.version for v in versions}

        for version_data in model_data.get("versions", []):
            version = version_data["version"]
            run_id = version_data.get("run_id")

            if version in existing_versions and self.skip_existing:
                self.stats["model_versions_skipped"] += 1
                print(f"    ⊘ Version {version} exists")
                continue

            if not self.dry_run and run_id:
                self._create_model_version(model_name, version_data, run_id)
            else:
                print(f"    ✓ Would create version {version}")
                self.stats["model_versions_created"] += 1

    def _create_model_version(
        self, model_name: str, version_data: Dict[str, Any], run_id: str
    ) -> None:
        try:
            mv = self.client.create_model_version(
                model_name,
                f"runs:/{run_id}/model",
                run_id=run_id,
                tags=version_data.get("tags", {}),
                description=version_data.get("description"),
            )

            current_stage = version_data.get("current_stage")
            if current_stage and current_stage != "None":
                self.client.transition_model_version_stage(
                    model_name, mv.version, current_stage
                )

            print(f"    ✓ Created version {mv.version}")
            self.stats["model_versions_created"] += 1
        except Exception as e:  # pylint: disable=broad-except
            print(
                f"    ⚠ Warning: Could not create version {version_data['version']}: {e}"
            )
            self.stats["model_versions_skipped"] += 1

    def _print_summary(self) -> None:
        print("\n" + "=" * 70)
        if self.dry_run:
            print("🔍 Dry Run Summary (No changes made)")
        else:
            print("✅ Import Complete!")
        print("=" * 70)
        print(
            f"📦 Experiments: {self.stats['experiments_created']} created, "
            f"{self.stats['experiments_skipped']} skipped"
        )
        print(
            f"🏃 Runs: {self.stats['runs_created']} created, "
            f"{self.stats['runs_skipped']} skipped"
        )
        print(f"🎯 Models: {self.stats['models_created']} created")
        print(
            f"📦 Model Versions: {self.stats['model_versions_created']} created, "
            f"{self.stats['model_versions_skipped']} skipped"
        )
        print(
            f"📁 Artifacts: {self.stats['artifacts_uploaded']} uploaded, "
            f"{self.stats['artifacts_skipped']} skipped"
        )

    def _cleanup(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def main() -> None:
    """Main entry point."""
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
        MLflowImporter(
            args.backup_file, skip_existing=not args.overwrite, dry_run=args.dry_run
        ).import_data()
        sys.exit(0)
    except Exception as e:  # pylint: disable=broad-except
        print(f"\n❌ Import failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

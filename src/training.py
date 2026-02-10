"""
Training module with comprehensive MLflow integration.

Features:
- Configurable YOLO architecture (supports yolo11n/s/m/l/x, extensible to future versions)
- Complete experiment tracking (all hyperparameters, metrics, artifacts)
- Model registry integration
- Extensive logging and visualization
"""

import os
import time
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import yaml
import torch
from ultralytics import YOLO
import pandas as pd


class YOLOTrainer:
    """YOLO training with comprehensive MLflow tracking."""

    def __init__(
        self,
        data_yaml_path: str,
        training_config: dict[str, Any],
        mlflow_tracking_uri: Optional[str] = None,
    ):
        """
        Initialize YOLO trainer.

        Args:
            data_yaml_path: Path to data.yaml file
            training_config: Training configuration dictionary
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.data_yaml_path = data_yaml_path
        self.config = training_config

        # Set MLflow tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

        # Extract configuration
        self.model_config = training_config["model"]
        self.dataset_config = training_config["dataset"]
        self.training_params = training_config["training"]
        self.augmentation = training_config["augmentation"]
        self.mlflow_config = training_config["mlflow"]
        self.device_config = training_config.get("device", {})

        # Model architecture
        self.architecture = self.model_config["architecture"]

        # Output directory
        self.output_dir = Path(training_config["paths"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.run = None

    def _get_model_path(self) -> str:
        """
        Get the model path based on architecture.
        Extensible for future YOLO versions.

        Returns:
            Model identifier for Ultralytics
        """
        architecture = self.architecture.lower()

        # Map architecture to model path
        # Easily extensible for future versions like yolo26
        model_map = {
            # YOLOv11 models
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolo11l": "yolo11l.pt",
            "yolo11x": "yolo11x.pt",
            # YOLOv26 models
            "yolo26n": "yolo26n.pt",
            "yolo26s": "yolo26s.pt",
            "yolo26m": "yolo26m.pt",
            "yolo26l": "yolo26l.pt",
            "yolo26x": "yolo26x.pt",
        }

        if architecture not in model_map:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Supported: {list(model_map.keys())}"
            )

        model_path = model_map[architecture]

        # If not using pretrained, return just the architecture name
        if not self.model_config.get("pretrained", True):
            return architecture + ".yaml"  # From scratch

        return model_path

    def _prepare_training_args(self) -> dict[str, Any]:
        """
        Prepare training arguments for Ultralytics YOLO.

        Returns:
            Dictionary of training arguments
        """
        args = {
            # Data
            "data": self.data_yaml_path,
            "imgsz": self.dataset_config["img_size"],
            # Training
            "epochs": self.training_params["epochs"],
            "batch": self.training_params["batch_size"],
            "patience": self.training_params["patience"],
            "workers": self.training_params["workers"],
            # Optimizer
            "lr0": self.training_params["lr0"],
            "lrf": self.training_params["lrf"],
            "momentum": self.training_params["momentum"],
            "weight_decay": self.training_params["weight_decay"],
            # Warmup
            "warmup_epochs": self.training_params["warmup_epochs"],
            "warmup_momentum": self.training_params["warmup_momentum"],
            "warmup_bias_lr": self.training_params["warmup_bias_lr"],
            # Augmentation
            "hsv_h": self.augmentation["hsv_h"],
            "hsv_s": self.augmentation["hsv_s"],
            "hsv_v": self.augmentation["hsv_v"],
            "degrees": self.augmentation["degrees"],
            "translate": self.augmentation["translate"],
            "scale": self.augmentation["scale"],
            "shear": self.augmentation["shear"],
            "perspective": self.augmentation["perspective"],
            "flipud": self.augmentation["flipud"],
            "fliplr": self.augmentation["fliplr"],
            "mosaic": self.augmentation["mosaic"],
            "mixup": self.augmentation["mixup"],
            "copy_paste": self.augmentation["copy_paste"],
            # Seed (Critical for reproducibility)
            "seed": self.dataset_config["seed"],
            # Other
            "amp": self.training_params["amp"],
            "close_mosaic": self.training_params["close_mosaic"],
            "project": str(self.output_dir),
            "name": f"{self.architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "exist_ok": True,
            "pretrained": self.model_config.get("pretrained", True),
            "verbose": True,
            "save": True,
            "save_period": -1,  # Save only last and best
        }

        # Device
        if self.device_config.get("device"):
            args["device"] = self.device_config["device"]

        return args

    def _log_all_parameters(self, training_args: dict[str, Any]) -> None:
        """
        Log ALL parameters to MLflow for complete experiment tracking.

        Args:
            training_args: Training arguments dictionary
        """
        print("\n📊 Logging parameters to MLflow...")

        # Model architecture
        mlflow.log_param("model_architecture", self.architecture)
        mlflow.log_param("model_pretrained", self.model_config.get("pretrained", True))

        # Dataset configuration
        mlflow.log_param("img_size", self.dataset_config["img_size"])
        mlflow.log_param("split_train", self.dataset_config["split_ratios"]["train"])
        mlflow.log_param("split_val", self.dataset_config["split_ratios"]["val"])
        mlflow.log_param("split_test", self.dataset_config["split_ratios"]["test"])
        mlflow.log_param("seed", self.dataset_config["seed"])

        # Training hyperparameters
        for key, value in self.training_params.items():
            mlflow.log_param(f"train_{key}", value)

        # Augmentation settings
        for key, value in self.augmentation.items():
            mlflow.log_param(f"aug_{key}", value)

        # System info
        mlflow.log_param("config_device", training_args.get("device", "auto"))
        mlflow.log_param("cuda_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            mlflow.log_param("gpu_count", torch.cuda.device_count())

        # Log config files as artifacts
        with open("temp_training_config.yaml", "w") as f:
            yaml.dump(self.config, f)
        mlflow.log_artifact("temp_training_config.yaml", "configs")
        os.remove("temp_training_config.yaml")

        print("✓ All parameters logged")

    def _log_training_metrics(self, results_csv: Path) -> None:
        """
        Log all training metrics from results.csv to MLflow.

        Args:
            results_csv: Path to results.csv file
        """
        print("\n📈 Logging training metrics...")

        if not results_csv.exists():
            print("⚠ results.csv not found, skipping metric logging")
            return

        # Load results
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # Remove whitespace

        # Log metrics for each epoch
        for idx, row in df.iterrows():
            epoch = int(row["epoch"]) if "epoch" in row else idx

            # Log all available metrics
            metrics_to_log = {
                # Training metrics
                "train/box_loss": "train/box_loss",
                "train/cls_loss": "train/cls_loss",
                "train/dfl_loss": "train/dfl_loss",
                # Validation metrics
                "metrics/precision(B)": "val/precision",
                "metrics/recall(B)": "val/recall",
                "metrics/mAP50(B)": "val/mAP50",
                "metrics/mAP50-95(B)": "val/mAP50-95",
                # Learning rate
                "lr/pg0": "lr/pg0",
                "lr/pg1": "lr/pg1",
                "lr/pg2": "lr/pg2",
            }

            for csv_col, mlflow_name in metrics_to_log.items():
                if csv_col in row:
                    mlflow.log_metric(mlflow_name, float(row[csv_col]), step=epoch)

        print(f"✓ Logged metrics for {len(df)} epochs")

    def _log_training_artifacts(self, run_dir: Path) -> None:
        """
        Log all training artifacts to MLflow.

        Args:
            run_dir: Path to YOLO training run directory
        """
        print("\n📦 Logging artifacts...")

        # Log confusion matrix
        confusion_matrix = run_dir / "confusion_matrix.png"
        if confusion_matrix.exists():
            mlflow.log_artifact(str(confusion_matrix), "visualizations")

        # Log PR curve
        pr_curve = run_dir / "PR_curve.png"
        if pr_curve.exists():
            mlflow.log_artifact(str(pr_curve), "visualizations")

        # Log F1 curve
        f1_curve = run_dir / "F1_curve.png"
        if f1_curve.exists():
            mlflow.log_artifact(str(f1_curve), "visualizations")

        # Log training curves
        results_png = run_dir / "results.png"
        if results_png.exists():
            mlflow.log_artifact(str(results_png), "visualizations")

        # Log validation predictions
        val_batch_labels = list(run_dir.glob("val_batch*_labels.jpg"))
        val_batch_pred = list(run_dir.glob("val_batch*_pred.jpg"))

        for img_path in val_batch_labels[:5]:  # Log first 5
            mlflow.log_artifact(str(img_path), "validation_samples")

        for img_path in val_batch_pred[:5]:
            mlflow.log_artifact(str(img_path), "validation_samples")

        # Log results CSV
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            mlflow.log_artifact(str(results_csv), "metrics")

        # Log model weights (batch upload)
        weights_dir = run_dir / "weights"
        if weights_dir.exists():
            print(f"  Uploading weights directory: {weights_dir}")
            mlflow.log_artifacts(str(weights_dir), "weights")
        else:
            print("⚠ Weights directory not found")

        print("✓ All artifacts logged")

    def _register_model(
        self,
        best_model_path: Path,
        final_metrics: dict[str, float],
        model_name_base: str,
    ) -> None:
        """
        Register model to MLflow Model Registry.

        Args:
            best_model_path: Path to best model weights
            final_metrics: Dictionary of final metrics
            model_name_base: Dynamic name for the model (e.g., yolo26n_20260205_120000)
        """
        if not self.mlflow_config.get("auto_register", True):
            print("\n⏭ Auto-registration disabled, skipping model registration")
            return

        print(f"\n🏷 Registering model '{model_name_base}' to MLflow Model Registry...")

        try:
            # Get the current run info
            run = mlflow.active_run()
            if not run:
                print("⚠ No active MLflow run, skipping registration")
                return

            run_id = run.info.run_id
            client = mlflow.tracking.MlflowClient()

            # CRITICAL: Ensure the artifact was logged correctly
            # Explicitly log the artifact to ensuring it exists in MinIO
            mlflow.log_artifact(str(best_model_path), "weights")
            print("  ✓ Verified: Model weights uploaded to 'weights/best.pt'")

            # Use dynamic model name
            model_name = model_name_base

            # Ensure registered model exists
            try:
                client.create_registered_model(model_name)
                print(f"  ✓ Created registered model: {model_name}")
            except Exception:
                # Model already exists
                pass

            # Create model version using direct API
            # Source must match where we logged it above: "runs:/<run_id>/weights/best.pt"
            source_uri = f"runs:/{run_id}/weights/best.pt"

            model_version = client.create_model_version(
                name=model_name, source=source_uri, run_id=run_id
            )

            latest_version = model_version.version
            print(f"  ✓ Created model version: {latest_version}")

            # Add description
            try:
                client.update_model_version(
                    name=model_name,
                    version=latest_version,
                    description=f"YOLOv11 Finger Counting Model - {self.architecture}\n"
                    f"Run ID: {run_id}\n"
                    f"mAP@50-95: {final_metrics.get('mAP50-95', 'N/A'):.4f}",
                )
            except Exception as e:  # pylint: disable=broad-except
                print(f"  ⚠ Description update failed (non-critical): {e}")

            # Promote to stage if configured
            auto_promote_stage = self.mlflow_config.get("auto_promote_stage")
            if auto_promote_stage:
                try:
                    client.transition_model_version_stage(
                        name=model_name,
                        version=latest_version,
                        stage=auto_promote_stage,
                        archive_existing_versions=True,
                    )
                    print(f"  ✓ Model promoted to '{auto_promote_stage}' stage")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"  ⚠ Promotion failed: {e}")

            print(f"✅ Model successfully registered: {model_name} v{latest_version}")

        except Exception as e:  # pylint: disable=broad-except
            print(f"⚠ Model registration failed: {e}")
            import traceback

            traceback.print_exc()

    def _ensure_experiment_active(self) -> None:
        """
        Ensure the MLflow experiment is active and not deleted.
        Restores deleted experiments if found.
        """
        experiment_name = self.mlflow_config["experiment_name"]
        client = MlflowClient()

        try:
            # Check if experiment exists
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment:
                if experiment.lifecycle_stage == "deleted":
                    print(f"♻ Restoring deleted experiment: {experiment_name}")
                    client.restore_experiment(experiment.experiment_id)
        except Exception as e:  # pylint: disable=broad-except
            print(f"⚠ Note on experiment setup: {e}")

        # Set as active (creates if doesn't exist)
        mlflow.set_experiment(experiment_name)

        # CRITICAL for Ultralytics: Set env var so it picks up the right experiment
        # otherwise it defaults to 'project' name (which is 'experiments')
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    def train(self) -> dict[str, Any]:
        """
        Execute complete training pipeline with MLflow tracking.

        Returns:
            Dictionary with training results
        """
        print("\n" + "=" * 70)
        print("🚀 Starting YOLOv11 Training with MLflow Tracking")
        print("=" * 70)

        # Ensure experiment exists and is active (handle deleted state)
        self._ensure_experiment_active()

        # Start MLflow run
        run_name = f"{self.mlflow_config['run_name_prefix']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name) as run:
            self.run = run

            # Set tags
            for key, value in self.mlflow_config["tags"].items():
                mlflow.set_tag(key, value)

            mlflow.set_tag("start_time", datetime.now().isoformat())

            # Get model path
            model_path = self._get_model_path()
            print(f"\n📦 Loading model: {model_path}")

            # Initialize model
            self.model = YOLO(model_path)

            # Prepare training arguments
            training_args = self._prepare_training_args()

            # Log all parameters
            self._log_all_parameters(training_args)

            # Force Ultralytics to use this specific run instead of creating a new one (https://mlflow.org/docs/latest/api_reference/python_api/mlflow.environment_variables.html)
            os.environ["MLFLOW_RUN_ID"] = run.info.run_id

            # Start training
            print(
                f"\n🏋️ Training {self.architecture} for {self.training_params['epochs']} epochs..."
            )
            start_time = time.time()

            results = self.model.train(**training_args)

            training_duration = time.time() - start_time

            # Log training duration
            mlflow.log_metric("training_duration_seconds", training_duration)
            mlflow.log_metric("training_duration_hours", training_duration / 3600)

            # Get run directory
            run_dir = Path(results.save_dir)

            # Log training metrics
            results_csv = run_dir / "results.csv"
            self._log_training_metrics(results_csv)

            # Log artifacts
            self._log_training_artifacts(run_dir)

            # Get final metrics
            final_metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
            }

            # Log final metrics
            for key, value in final_metrics.items():
                mlflow.log_metric(f"final_{key}", value)

            # Register model
            best_model_path = run_dir / "weights" / "best.pt"

            # Use the run_name as the base for the model name
            # run_name is already: {prefix}_{YYYYMMDD_HHMMSS}
            self._register_model(best_model_path, final_metrics, run_name)

            mlflow.set_tag("end_time", datetime.now().isoformat())
            mlflow.set_tag("status", "completed")

            print("\n" + "=" * 70)
            print("✅ Training Complete!")
            print("=" * 70)
            print("📊 Final Metrics:")
            for key, value in final_metrics.items():
                print(f"  - {key}: {value:.4f}")
            print(f"⏱ Training Duration: {training_duration / 3600:.2f} hours")
            print(f"🔗 MLflow Run: {run.info.run_id}")
            print("=" * 70)

            return {
                "run_id": run.info.run_id,
                "metrics": final_metrics,
                "duration": training_duration,
                "model_path": str(best_model_path),
            }

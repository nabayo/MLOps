"""
Evaluation module with comprehensive MLflow logging.

Features:
- Complete evaluation on test set
- Per-class metrics
- Extensive visualizations
- Error analysis
- Confidence calibration
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import torch


class YOLOEvaluator:
    """YOLO model evaluation with comprehensive MLflow tracking."""

    def __init__(
        self,
        model_path: str,
        data_yaml_path: str,
        mlflow_run_id: str = None
    ):
        """
        Initialize YOLO evaluator.

        Args:
            model_path: Path to trained model weights
            data_yaml_path: Path to data.yaml file
            mlflow_run_id: Existing MLflow run ID to log to
        """
        self.model_path = Path(model_path)
        self.data_yaml_path = data_yaml_path
        self.mlflow_run_id = mlflow_run_id

        # Load model
        print(f"📦 Loading model from {model_path}")
        self.model = YOLO(str(model_path))

    def evaluate(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.

        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "=" * 70)
        print("🧪 Starting Model Evaluation")
        print("=" * 70)

        # Check if we're already in a run context
        existing_run = mlflow.active_run()
        should_end_run = False
        
        if existing_run and self.mlflow_run_id:
            # If there's an active run and it matches our target, use it
            if existing_run.info.run_id == self.mlflow_run_id:
                print(f"✓ Using existing active run: {self.mlflow_run_id}")
                run = existing_run
            else:
                # Different run is active, end it and start ours
                print(f"⚠ Ending unrelated run {existing_run.info.run_id}")
                mlflow.end_run()
                
                # Get the experiment ID from the target run to avoid mismatch
                client = mlflow.tracking.MlflowClient()
                target_run_info = client.get_run(self.mlflow_run_id)
                target_experiment_id = target_run_info.info.experiment_id
                
                # Set the experiment before starting the run
                mlflow.set_experiment(experiment_id=target_experiment_id)
                
                run = mlflow.start_run(run_id=self.mlflow_run_id)
                should_end_run = True
        elif existing_run and not self.mlflow_run_id:
            # Active run exists but we don't have a target ID, use the active one
            print(f"✓ Using existing active run: {existing_run.info.run_id}")
            run = existing_run
        elif self.mlflow_run_id:
            # No active run but we have a target ID
            # Get the experiment ID from the target run to avoid mismatch
            client = mlflow.tracking.MlflowClient()
            target_run_info = client.get_run(self.mlflow_run_id)
            target_experiment_id = target_run_info.info.experiment_id
            
            # Set the experiment before starting the run
            mlflow.set_experiment(experiment_id=target_experiment_id)
            
            run = mlflow.start_run(run_id=self.mlflow_run_id)
            should_end_run = True
        else:
            # No active run and no target ID, create new
            run = mlflow.start_run()
            should_end_run = True

        try:
            # Run validation on test set
            print("\n📊 Running evaluation on test set...")
            results = self.model.val(
                data=self.data_yaml_path,
                split='test',
                save_json=True,
                save_hybrid=True,
                conf=0.001,  # Low confidence for complete analysis
                iou=0.6,
                plots=True
            )

            # Extract metrics
            metrics = {
                'test/mAP50': results.box.map50,
                'test/mAP50-95': results.box.map,
                'test/precision': results.box.mp,
                'test/recall': results.box.mr,
                'test/f1': 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-6)
            }

            # Log metrics
            print("\n📈 Logging evaluation metrics...")
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
                print(f"  - {key}: {value:.4f}")

            # Per-class metrics
            if hasattr(results.box, 'ap_class_index'):
                print("\n📊 Logging per-class metrics...")
                class_names = results.names

                for idx, class_id in enumerate(results.box.ap_class_index):
                    class_name = class_names[int(class_id)]

                    # Log per-class AP
                    if hasattr(results.box, 'ap'):
                        if results.box.ap.ndim > 1:
                            ap50 = results.box.ap[idx, 0]  # AP@0.5
                            ap50_95 = results.box.ap[idx].mean()  # AP@0.5:0.95
                        else:
                            # Handle 1D array case (when only one class or different shape)
                            ap50 = results.box.ap[0]  # AP@0.5
                            ap50_95 = results.box.ap.mean()  # AP@0.5:0.95

                        mlflow.log_metric(f"test/ap50_class_{class_name}", float(ap50))
                        mlflow.log_metric(f"test/ap50-95_class_{class_name}", float(ap50_95))

            # Log visualizations
            self._log_visualizations(results)

            # Create additional analysis
            self._create_error_analysis(results)

            print("\n" + "=" * 70)
            print("✅ Evaluation Complete!")
            print("=" * 70)

            return {
                'metrics': metrics,
                'run_id': run.info.run_id
            }
        finally:
            # Only end the run if we started it
            if should_end_run:
                mlflow.end_run()

    def _log_visualizations(self, results) -> None:
        """
        Log all evaluation visualizations.

        Args:
            results: YOLO validation results
        """
        print("\n📦 Logging visualizations...")

        # Get save directory
        save_dir = Path(results.save_dir)

        # Log confusion matrix
        confusion_matrix = save_dir / 'confusion_matrix_normalized.png'
        if confusion_matrix.exists():
            mlflow.log_artifact(str(confusion_matrix), 'test_visualizations')

        # Log PR curve
        pr_curve = save_dir / 'PR_curve.png'
        if pr_curve.exists():
            mlflow.log_artifact(str(pr_curve), 'test_visualizations')

        # Log F1 curve
        f1_curve = save_dir / 'F1_curve.png'
        if f1_curve.exists():
            mlflow.log_artifact(str(f1_curve), 'test_visualizations')

        # Log P curve
        p_curve = save_dir / 'P_curve.png'
        if p_curve.exists():
            mlflow.log_artifact(str(p_curve), 'test_visualizations')

        # Log R curve
        r_curve = save_dir / 'R_curve.png'
        if r_curve.exists():
            mlflow.log_artifact(str(r_curve), 'test_visualizations')

        print("✓ Visualizations logged")

    def _create_error_analysis(self, results) -> None:
        """
        Create and log error analysis visualizations.

        Args:
            results: YOLO validation results
        """
        print("\n🔍 Creating error analysis...")

        # Create confidence distribution plot
        self._plot_confidence_distribution()

        print("✓ Error analysis complete")

    def _plot_confidence_distribution(self) -> None:
        """Plot and log confidence score distribution."""
        try:
            # Run prediction to get confidence scores
            results = self.model.predict(
                source=self.data_yaml_path,
                save=False,
                conf=0.001
            )

            # Collect confidences
            confidences = []
            for result in results:
                if result.boxes is not None:
                    confidences.extend(result.boxes.conf.cpu().numpy().tolist())

            if len(confidences) > 0:
                # Create plot
                plt.figure(figsize=(10, 6))
                plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
                plt.xlabel('Confidence Score')
                plt.ylabel('Frequency')
                plt.title('Prediction Confidence Distribution')
                plt.grid(True, alpha=0.3)

                # Save and log
                conf_dist_path = 'confidence_distribution.png'
                plt.savefig(conf_dist_path, dpi=150, bbox_inches='tight')
                plt.close()

                mlflow.log_artifact(conf_dist_path, 'test_visualizations')
                os.remove(conf_dist_path)

        except Exception as e:
            print(f"⚠ Could not create confidence distribution: {e}")

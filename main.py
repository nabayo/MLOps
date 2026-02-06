"""
MLOps Pipeline Orchestrator

Main entry point for training, evaluation, and serving.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.config import load_config
from src.picsellia import load_data
from src.data_preparation import DataPreparation
from src.training import YOLOTrainer
from src.evaluation import YOLOEvaluator


def load_training_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_training(args: argparse.Namespace) -> None:
    """
    Run complete training pipeline.

    Args:
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("🚀 MLOps Training Pipeline")
    print("=" * 80)

    # Load configurations
    config = load_config()
    training_config = load_training_config(args.training_config)

    # Step 1: Load data from Picsellia
    print("\n📦 Step 1: Loading dataset from Picsellia...")
    dataset_loader = load_data(config)
    dataset_path = dataset_loader.dataset_path
    print(f"✓ Dataset loaded: {dataset_path}")

    # Step 2: Prepare data
    print("\n🔧 Step 2: Preparing data...")
    data_prep = DataPreparation(
        dataset_path=str(dataset_path),
        config=config,
        training_config=training_config
    )
    data_yaml_path = data_prep.prepare()
    print(f"✓ Data prepared: {data_yaml_path}")

    # Step 3: Train model
    print("\n🏋️ Step 3: Training model...")
    trainer = YOLOTrainer(
        data_yaml_path=data_yaml_path,
        training_config=training_config,
        mlflow_tracking_uri=os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    )
    train_results = trainer.train()
    print(f"✓ Training completed!")
    print(f"   Run ID: {train_results['run_id']}")
    print(f"   Duration: {train_results['duration']/3600:.2f}h")

    # Step 4: Evaluate model (if requested)
    if args.evaluate:
        print("\n🧪 Step 4: Evaluating model...")
        evaluator = YOLOEvaluator(
            model_path=train_results['model_path'],
            data_yaml_path=data_yaml_path,
            mlflow_run_id=train_results['run_id']
        )
        eval_results = evaluator.evaluate()
        print(f"✓ Evaluation completed!")

    print("\n" + "=" * 80)
    print("✅ Pipeline Complete!")
    print("=" * 80)
    print(f"\n🔗 View results in MLflow UI: {os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")


def run_evaluation(args: argparse.Namespace) -> None:
    """
    Run evaluation on existing model.

    Args:
        args: Command-line arguments
    """
    print("\n" + "=" * 80)
    print("🧪 Model Evaluation")
    print("=" * 80)

    # Load configuration
    config = load_config()
    training_config = load_training_config(args.training_config)

    # Get dataset
    dataset_loader = load_data(config)
    dataset_path = dataset_loader.dataset_path

    # Prepare data if needed
    data_yaml_path = Path(dataset_path) / "yolo_format" / "data.yaml"
    if not data_yaml_path.exists():
        print("\n⚠ Data not prepared, running preparation...")
        data_prep = DataPreparation(
            dataset_path=str(dataset_path),
            config=config,
            training_config=training_config
        )
        data_yaml_path = data_prep.prepare()

    # Run evaluation
    evaluator = YOLOEvaluator(
        model_path=args.model,
        data_yaml_path=str(data_yaml_path),
        mlflow_run_id=args.run_id
    )
    results = evaluator.evaluate()

    print("\n✅ Evaluation Complete!")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="MLOps Pipeline for YOLOv11 Finger Counting",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Training mode
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    train_parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation after training'
    )

    # Evaluation mode
    eval_parser = subparsers.add_parser('eval', help='Evaluate an existing model')
    eval_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model weights (.pt file)'
    )
    eval_parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )
    eval_parser.add_argument(
        '--run-id',
        type=str,
        default=None,
        help='MLflow run ID to log results to'
    )

    # Serve mode (handled by separate serving/api.py)
    serve_parser = subparsers.add_parser('serve', help='Start serving API (use serving/api.py instead)')

    # Clean mode
    clean_parser = subparsers.add_parser('clean', help='Clean dataset directory')
    clean_parser.add_argument(
        '--training-config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training configuration file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute based on mode
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'eval':
        run_evaluation(args)
    elif args.mode == 'serve':
        print("❌ Use 'python serving/api.py' or 'docker-compose up serving' to start the serving API")
        sys.exit(1)
    elif args.mode == 'clean':
        print("\n" + "=" * 80)
        print("🧹 Cleaning Dataset")
        print("=" * 80)
        
        # Load configuration
        config = load_config()
        
        # Get dataset path using loader utils
        from src.dataset_loader import get_dataset_download_path
        dataset_path = get_dataset_download_path(config)
        
        if os.path.exists(dataset_path):
            import shutil
            try:
                print(f"Removing dataset directory: {dataset_path}")
                shutil.rmtree(dataset_path)
                print("✓ Dataset removed successfully")
            except Exception as e:
                print(f"❌ Error removing dataset: {e}")
        else:
            print(f"⚠ Dataset directory not found: {dataset_path}")
            
        print("\n✅ Clean Complete!")
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
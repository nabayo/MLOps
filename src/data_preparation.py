"""
Data preparation module for YOLOv11 training.

This module handles:
- Dataset validation
- YOLO format conversion from Picsellia
- Configurable train/val/test splitting
- Dynamic data.yaml generation for Ultralytics
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm


class DataPreparation:
    """Handles data preparation for YOLO training."""

    def __init__(
        self,
        dataset_path: str,
        config: Dict[str, Any],
        training_config: Dict[str, Any]
    ):
        """
        Initialize data preparation.

        Args:
            dataset_path: Path to downloaded dataset from Picsellia
            config: Dataset configuration (from config.yaml)
            training_config: Training configuration (from training_config.yaml)
        """
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.training_config = training_config

        # Extract split ratios
        self.split_ratios = training_config['dataset']['split_ratios']
        self.seed = training_config['dataset']['seed']

        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Output paths
        self.output_path = self.dataset_path / "yolo_format"
        self.images_path = self.output_path / "images"
        self.labels_path = self.output_path / "labels"

    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate the dataset structure and integrity.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check if dataset path exists
        if not self.dataset_path.exists():
            errors.append(f"Dataset path does not exist: {self.dataset_path}")
            return False, errors

        # Check for annotations file
        annotations_path = self.dataset_path / "annotations.json"
        if not annotations_path.exists():
            errors.append(f"Annotations file not found: {annotations_path}")
            return False, errors

        # Load and validate annotations
        try:
            with open(annotations_path, 'r') as f:
                annotations = json.load(f)

            if not isinstance(annotations, dict):
                errors.append("Annotations file must contain a JSON object")

        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in annotations file: {e}")
            return False, errors

        print(f"✓ Dataset validation passed: {self.dataset_path}")
        return True, errors

    def convert_picsellia_to_yolo(self) -> Dict[str, List[str]]:
        """
        Convert Picsellia format annotations to YOLO format.

        Picsellia format: Dict[image_id, List[annotations with rectangles]]
        YOLO format: <class_id> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]

        Returns:
            Dictionary with 'images' and 'labels' lists of created files
        """
        print("\n🔄 Converting Picsellia format to YOLO format...")

        # Load annotations
        annotations_path = self.dataset_path / "annotations.json"
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)

        # Create output directories
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Parse annotations and convert
        image_files = []
        label_files = []
        class_names = set()

        # Find all image files in the dataset directory
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        available_images = {}
        for file_path in self.dataset_path.iterdir():
            if file_path.suffix in valid_extensions:
                available_images[file_path.stem] = file_path

        print(f"Found {len(available_images)} images in dataset directory")
        print(f"Found {len(annotations_data)} annotated images in annotations.json")

        # Build label mapping from annotations
        label_to_id = {}
        for image_id, ann_list in annotations_data.items():
            for ann in ann_list:
                for rect in ann.get('rectangles', []):
                    label = rect.get('label')
                    if label and label not in label_to_id:
                        class_names.add(label)

        # Create sorted label mapping (consistent across runs)
        class_names_sorted = sorted(class_names)
        label_to_id = {label: idx for idx, label in enumerate(class_names_sorted)}

        print(f"Detected classes: {class_names_sorted}")

        # Check if YOLO labels were already exported by SDK
        # SDK usually puts them in a zip file/subdirectory
        import zipfile
        
        # 1. Search for any zip file containing "YOLO" or just any zip in the dataset path (recursive)
        found_zip_files = list(self.dataset_path.rglob("*.zip"))
        
        if found_zip_files:
            print(f"found zip files: {found_zip_files}")
            for zip_file in found_zip_files:
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        # Extract to dataset path
                        print(f"Extracting {zip_file}...")
                        zip_ref.extractall(self.dataset_path)
                except Exception as e:
                    print(f"Warning: Failed to extract {zip_file}: {e}")
        
        # 2. Search for any .txt file in the download path (recursive now, to catch extracted files)
        sdk_exported_labels = list(self.dataset_path.rglob("*.txt"))
        
        # Filter out classes.txt or non-label files if possible, but usually safe
        sdk_exported_labels = [f for f in sdk_exported_labels if f.name != 'classes.txt']

        # If we found many txt files, assume they are the labels
        use_sdk_labels = len(sdk_exported_labels) > len(available_images) * 0.5
        
        if use_sdk_labels:
            print("✓ Found SDK-exported YOLO labels. Skipping manual conversion.")
            converted_count = 0
            
            for img_stem, img_path in tqdm(available_images.items(), desc="Processing SDK labels"):
                # Find matching txt file
                # SDK name matches image name usually
                label_path = self.dataset_path / (img_stem + ".txt")
                
                if not label_path.exists():
                    # Try looking for UUID match if filename match fails?
                    # For now assume Picsellia SDK matched them correctly
                    continue
                    
                # Copy image to YOLO directory
                dst_img = self.images_path / img_path.name
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_img)
                image_files.append(str(dst_img))
                
                # Copy label to YOLO directory
                label_filename = img_path.stem + '.txt'
                dst_label = self.labels_path / label_filename
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label_path, dst_label)
                label_files.append(str(dst_label))
                
                converted_count += 1
                
            print(f"✓ Processed {converted_count} images using SDK labels")
            
            # Use class names from file if available, else use inferred
            classes_file = self.dataset_path / "classes.txt"
            if classes_file.exists():
                with open(classes_file, 'r') as f:
                    self.class_names = [line.strip() for line in f.readlines() if line.strip()]
            else:
                 self.class_names = class_names_sorted if class_names_sorted else ["finger-1", "finger-2", "finger-3", "finger-4", "finger-5"]

            return {
                'images': image_files,
                'labels': label_files
            }

        # Process each annotated image (Manual Fallback)
        converted_count = 0
        skipped_count = 0
        clamped_count = 0

        for image_id, ann_list in tqdm(annotations_data.items(), desc="Converting annotations"):
            # Try to find the corresponding image file
            # Picsellia uses UUIDs as image_id, but actual filename may differ
            # We need to match by looking for images in the directory

            # Try to find matching image file
            image_path = None
            for img_stem, img_path in available_images.items():
                # Simple heuristic: check if UUID is in filename or vice versa
                if image_id in img_path.name or img_stem == image_id:
                    image_path = img_path
                    break
            
            # If UUID matching failed, use first available image (assumes order)
            # This is a fallback - better approach would be to use Picsellia SDK properly
            if image_path is None and available_images:
                # Use alphabetically first unused image
                used_images = {Path(f).stem for f in image_files}
                for img_stem in sorted(available_images.keys()):
                    if img_stem not in used_images:
                        image_path = available_images[img_stem]
                        break
            
            if image_path is None:
                skipped_count += 1
                continue

            # Get image dimensions
            try:
                with Image.open(image_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Warning: Could not read image {image_path}: {e}")
                skipped_count += 1
                continue

            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in ann_list:
                # Only process ACCEPTED annotations
                if ann.get('status') != 'ACCEPTED':
                    continue

                for rect in ann.get('rectangles', []):
                    # Get bounding box (Picsellia format: x, y, w, h in pixels)
                    x = rect.get('x', 0)
                    y = rect.get('y', 0)
                    w = rect.get('w', 0)
                    h = rect.get('h', 0)
                    label = rect.get('label')

                    if w <= 0 or h <= 0 or label not in label_to_id:
                        continue

                    # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    norm_width = w / img_width
                    norm_height = h / img_height
                    
                    # Check if clamping is needed
                    is_invalid = (
                        x_center > 1.0 or y_center > 1.0 or 
                        norm_width > 1.0 or norm_height > 1.0 or
                        x_center < 0 or y_center < 0
                    )
                    
                    if is_invalid:
                        clamped_count += 1
                        
                        # Clamp width/height first
                        norm_width = min(max(norm_width, 0.0), 1.0)
                        norm_height = min(max(norm_height, 0.0), 1.0)
                        
                        # Clamp center
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        
                        # Use a small epsilon to avoid edge cases
                        x_center = min(x_center, 0.999999)
                        y_center = min(y_center, 0.999999)
                        norm_width = min(norm_width, 0.999999)
                        norm_height = min(norm_height, 0.999999)

                    # Get class ID
                    class_id = label_to_id[label]

                    yolo_annotations.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                    )

            # Save label file (even if empty - indicating negative example)
            label_filename = image_path.stem + '.txt'
            label_path = self.labels_path / label_filename
            label_path.parent.mkdir(parents=True, exist_ok=True)

            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))

            label_files.append(str(label_path))

            # Copy image to YOLO directory
            dst_img = self.images_path / image_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dst_img)
            image_files.append(str(dst_img))

            converted_count += 1

        print(f"✓ Converted {converted_count} images and {len(label_files)} labels")
        if skipped_count > 0:
            print(f"⚠ Skipped {skipped_count} images (missing files)")
        if clamped_count > 0:
            print(f"⚠ Clamped/Fixed {clamped_count} annotations that were out of bounds")

        # Store class names
        self.class_names = class_names_sorted if class_names_sorted else ["finger-1", "finger-2", "finger-3", "finger-4", "finger-5"]

        return {
            'images': image_files,
            'labels': label_files
        }

    def create_splits(
        self,
        image_files: List[str]
    ) -> Dict[str, List[str]]:
        """
        Create train/val/test splits with configurable ratios.

        Args:
            image_files: List of image file paths

        Returns:
            Dictionary with 'train', 'val', 'test' lists of image paths
        """
        print(f"\n📊 Creating dataset splits (train: {self.split_ratios['train']:.0%}, "
              f"val: {self.split_ratios['val']:.0%}, test: {self.split_ratios['test']:.0%})...")

        # Shuffle images
        random.shuffle(image_files)

        # Calculate split indices
        total = len(image_files)
        train_size = int(total * self.split_ratios['train'])
        val_size = int(total * self.split_ratios['val'])

        # Split
        train_images = image_files[:train_size]
        val_images = image_files[train_size:train_size + val_size]
        test_images = image_files[train_size + val_size:]

        print(f"✓ Split created: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

        return {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }

    def organize_split_directories(self, splits: Dict[str, List[str]]) -> None:
        """
        Organize images and labels into train/val/test directories.

        Args:
            splits: Dictionary with split names and image paths
        """
        print("\n📁 Organizing split directories...")

        for split_name, image_paths in splits.items():
            # Create directories
            split_images_dir = self.output_path / 'images' / split_name
            split_labels_dir = self.output_path / 'labels' / split_name
            split_images_dir.mkdir(parents=True, exist_ok=True)
            split_labels_dir.mkdir(parents=True, exist_ok=True)

            # Move files
            for img_path in tqdm(image_paths, desc=f"Organizing {split_name}"):
                img_path = Path(img_path)

                # Copy image
                dst_img = split_images_dir / img_path.name
                shutil.copy2(img_path, dst_img)

                # Copy corresponding label
                label_name = img_path.stem + '.txt'
                src_label = self.labels_path / label_name
                dst_label = split_labels_dir / label_name

                if src_label.exists():
                    shutil.copy2(src_label, dst_label)

        print("✓ Split directories organized")

    def generate_data_yaml(self) -> str:
        """
        Generate data.yaml file for Ultralytics YOLO training.

        Returns:
            Path to generated data.yaml file
        """
        print("\n📝 Generating data.yaml for Ultralytics...")

        # Prepare data configuration
        data_config = {
            'path': str(self.output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.class_names),
            'names': list(self.class_names)
        }

        # Save data.yaml
        data_yaml_path = self.output_path / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ data.yaml generated: {data_yaml_path}")
        print(f"  - Classes: {data_config['names']}")
        print(f"  - Number of classes: {data_config['nc']}")

        return str(data_yaml_path)

    def prepare(self) -> str:
        """
        Run complete data preparation pipeline.

        Returns:
            Path to generated data.yaml file
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Data Preparation Pipeline")
        print("=" * 60)

        # Step 1: Validate
        is_valid, errors = self.validate_dataset()
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {errors}")

        # Step 2: Convert to YOLO format
        converted_files = self.convert_picsellia_to_yolo()

        # Step 3: Create splits
        splits = self.create_splits(converted_files['images'])

        # Step 4: Organize directories
        self.organize_split_directories(splits)

        # Step 5: Generate data.yaml
        data_yaml_path = self.generate_data_yaml()

        print("\n" + "=" * 60)
        print("✅ Data Preparation Complete!")
        print("=" * 60)

        return data_yaml_path

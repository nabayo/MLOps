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
        
        # Handle different Picsellia annotation formats
        if 'labelmap' in annotations_data:
            # Extract class names
            labelmap = annotations_data.get('labelmap', {})
            class_to_id = {name: idx for idx, name in enumerate(sorted(labelmap.keys()))}
            class_names = sorted(labelmap.keys())
            
        # Get images and annotations
        images = annotations_data.get('images', [])
        if isinstance(annotations_data, dict) and 'annotations' in annotations_data:
            annotations = annotations_data['annotations']
        else:
            annotations = []
        
        # Create a mapping of image_id to annotations
        image_annotations = {}
        for ann in annotations:
            img_id = ann.get('image_id')
            if img_id not in image_annotations:
                image_annotations[img_id] = []
            image_annotations[img_id].append(ann)
        
        # Process each image
        for img_data in tqdm(images, desc="Converting annotations"):
            img_id = img_data.get('id')
            img_filename = img_data.get('filename') or img_data.get('file_name')
            
            if not img_filename:
                continue
            
            # Image dimensions
            img_width = img_data.get('width', 0)
            img_height = img_data.get('height', 0)
            
            # If dimensions not in metadata, read from image
            if img_width == 0 or img_height == 0:
                img_path = self.dataset_path / img_filename
                if img_path.exists():
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
            
            # Get annotations for this image
            img_anns = image_annotations.get(img_id, [])
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in img_anns:
                # Get bounding box (format varies)
                bbox = ann.get('bbox', [])
                if not bbox or len(bbox) != 4:
                    continue
                
                # Picsellia bbox: [x_min, y_min, width, height]
                x_min, y_min, box_width, box_height = bbox
                
                # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = (x_min + box_width / 2) / img_width
                y_center = (y_min + box_height / 2) / img_height
                norm_width = box_width / img_width
                norm_height = box_height / img_height
                
                # Get class name and ID
                category_id = ann.get('category_id', 0)
                class_id = category_id  # Use category_id directly as class_id
                
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                )
            
            # Save label file (even if empty)
            label_filename = Path(img_filename).stem + '.txt'
            label_path = self.labels_path / label_filename
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            label_files.append(str(label_path))
            
            # Copy image to YOLO directory
            src_img = self.dataset_path / img_filename
            dst_img = self.images_path / img_filename
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                image_files.append(str(dst_img))
        
        print(f"✓ Converted {len(image_files)} images and {len(label_files)} labels")
        
        # Store class names
        self.class_names = class_names if class_names else ["finger-1", "finger-2", "finger-3", "finger-4", "finger-5"]
        
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

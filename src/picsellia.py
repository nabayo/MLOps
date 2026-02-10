"""
This module is used to load the dataset from Picsellia.
"""

from typing import Any

import os
import json

from src.dataset_loader import (
    DatasetLoader,
    get_dataset_download_path,
    get_annotations_path,
)

from picsellia import Client as PicselliaClient
from picsellia import Dataset as PicselliaDataset
from picsellia import DatasetVersion as PicselliaDatasetVersion
from picsellia.types.enums import AnnotationFileType


def load_data(config: dict[str, Any]) -> DatasetLoader:
    """
    Load the dataset from Picsellia.

    Args:
        config: The configuration dictionary.

    Returns:
        The dataset loader.
    """

    # Define the download path
    download_path: str = get_dataset_download_path(config)

    # Define the annotations path
    annotations_path: str = get_annotations_path(config)

    # Download the dataset if not already downloaded
    if (
        not os.path.exists(download_path)
        or not os.listdir(download_path)
        or not os.path.exists(annotations_path)
    ):
        # connect to picsellia
        client: PicselliaClient = PicselliaClient(api_token=config["picsellia_token"])

        # Get the dataset
        dataset: PicselliaDataset = client.get_dataset(name=config["dataset_name"])

        # Get the datasetversion
        dataset_version: PicselliaDatasetVersion = dataset.get_version(
            version=config["dataset_version"]
        )

        # Create the directory
        os.makedirs(name=download_path, exist_ok=True)

        # Download the dataset (images)
        dataset_version.download(target_path=download_path)

        try:
            # Export YOLO annotations using the SDK, download a zip file
            print("Exporting YOLO annotations using SDK...")
            dataset_version.export_annotation_file(
                annotation_file_type=AnnotationFileType.YOLO, target_path=download_path
            )

        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: Failed to export YOLO annotations via SDK: {e}")
            print("Falling back to manual JSON download.")

            # Get the regular annotations (for fallback/validation)
            annotations: dict[str, Any] = dataset_version.load_annotations()

            # Save the annotations
            with open(annotations_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f)

    # Load the dataset
    dataset_loader: DatasetLoader = DatasetLoader(config)
    dataset_loader.load_data()

    # Return the dataset loader
    return dataset_loader

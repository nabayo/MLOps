from typing import Any

import os
import json

from src.dataset_loader import DatasetLoader, get_dataset_download_path, get_annotations_path

from picsellia import Client as PicselliaClient
from picsellia import Dataset as PicselliaDataset
from picsellia import DatasetVersion as PicselliaDatasetVersion


def load_data(config: dict[str, Any]) -> DatasetLoader:

    # Define the download path
    download_path: str = get_dataset_download_path(config)

    # Define the annotations path
    annotations_path: str = get_annotations_path(config)

    # Download the dataset if not already downloaded
    if not os.path.exists(download_path) or not os.listdir(download_path) or not os.path.exists(annotations_path):

        # connect to picsellia
        client: PicselliaClient = PicselliaClient(api_token=config["picsellia_token"])

        # Get the dataset
        dataset: PicselliaDataset = client.get_dataset(name=config["dataset_name"])

        # Get the datasetversion
        dataset_version: PicselliaDatasetVersion = dataset.get_version(version=config["dataset_version"])

        # Create the directory
        os.makedirs(name=download_path, exist_ok=True)

        # Download the dataset
        dataset_version.download(target_path=download_path)

        # Get the annotations
        annotations: dict[str, Any] = dataset_version.load_annotations()

        # Save the annotations
        with open(annotations_path, "w") as f:
            json.dump(annotations, f)

    # Load the dataset
    dataset_loader: DatasetLoader = DatasetLoader(config)
    dataset_loader.load_data()

    # Return the dataset loader
    return dataset_loader

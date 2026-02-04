from typing import Any

import os

from src.dataset_loader import DatasetLoader, get_dataset_download_path

from picsellia import Client as PicselliaClient
from picsellia import Dataset as PicselliaDataset
from picsellia import DatasetVersion as PicselliaDatasetVersion


def load_data(config: dict[str, Any]) -> DatasetLoader:

    # connect to picsellia
    client: PicselliaClient = PicselliaClient(api_token=config["picsellia_token"])

    # Get the dataset
    dataset: PicselliaDataset = client.get_dataset(name=config["dataset_name"])

    # Get the datasetversion
    dataset_version: PicselliaDatasetVersion = dataset.get_version(version=config["dataset_version"])

    # Define the download path
    download_path: str = get_dataset_download_path(config)

    # Download the dataset if not already downloaded
    if not os.path.exists(download_path):

        # Create the directory
        os.makedirs(download_path)

        # Download the dataset
        dataset_version.download(target_path=download_path)

    # Load the dataset
    dataset_loader: DatasetLoader = DatasetLoader(config)
    dataset_loader.load_data()

    # Return the dataset loader
    return dataset_loader

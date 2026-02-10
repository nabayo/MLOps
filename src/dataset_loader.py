"""
Dataset loader module.
"""

from typing import Any, Optional

import os

import numpy as np
from numpy.typing import NDArray


def get_dataset_download_path(config: dict[str, Any]) -> str:
    """
    Get the dataset download path.

    Args:
        config: The configuration dictionary.

    Returns:
        The dataset download path.
    """

    dataset_download_path: str = config["dataset_download_path"].replace(" ", "_")
    dataset_name: str = config["dataset_name"].replace(" ", "_")
    dataset_version: str = config["dataset_version"].replace(" ", "_")

    relative_path: str = os.path.join(
        dataset_download_path, dataset_name, dataset_version
    )

    absolute_path: str = os.path.abspath(relative_path)

    return absolute_path


def get_annotations_path(config: dict[str, Any]) -> str:
    """
    Get the annotations path.

    Args:
        config: The configuration dictionary.

    Returns:
        The annotations path.
    """

    dataset_download_path: str = config["dataset_download_path"].replace(" ", "_")
    dataset_name: str = config["dataset_name"].replace(" ", "_")
    dataset_version: str = config["dataset_version"].replace(" ", "_")

    relative_path: str = os.path.join(
        dataset_download_path, dataset_name, dataset_version, "annotations.json"
    )

    absolute_path: str = os.path.abspath(relative_path)

    return absolute_path


class DatasetLoader:
    """
    Dataset loader class.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize dataset loader.

        Args:
            config: The configuration dictionary.
        """

        # Config
        self.config: dict[str, Any] = config

        # Dataset path
        self.dataset_path: str = get_dataset_download_path(config)

        # Data
        self.x_train: Optional[NDArray[np.uint8]] = None
        self.y_train: Optional[NDArray[np.uint8]] = None
        self.x_val: Optional[NDArray[np.uint8]] = None
        self.y_val: Optional[NDArray[np.uint8]] = None
        self.x_test: Optional[NDArray[np.uint8]] = None
        self.y_test: Optional[NDArray[np.uint8]] = None

    def load_data(self) -> None:
        """
        Load the dataset.
        """

        # Load the dataset
        print("Loading data...")

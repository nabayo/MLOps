from typing import Any, Optional

import os

import numpy as np
from numpy.typing import NDArray


def get_dataset_download_path(config: dict[str, Any]) -> str:

    dataset_download_path: str = config["dataset_download_path"].replace(" ", "_")
    dataset_name: str = config["dataset_name"].replace(" ", "_")
    dataset_version: str = config["dataset_version"].replace(" ", "_")

    return os.path.join(dataset_download_path, dataset_name, dataset_version)


class DatasetLoader:

    def __init__(self, config: dict[str, Any]):

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

        # Load the dataset
        pass



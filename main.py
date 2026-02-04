from typing import Any

from src.config import load_config
from src.picsellia import load_data


def main():

    # Load config
    config: dict[str, Any] = load_config()

    # Load data
    dataset_loader: DatasetLoader = load_data(config)


if __name__ == "__main__":
    main()
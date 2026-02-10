"""
Config module.
"""

from typing import Any

import yaml


def load_config() -> dict[str, Any]:
    """
    Load the configuration from the YAML file.

    Returns:
        dict[str, Any]: The configuration dictionary.
    """

    # load config from yaml file
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # load token from file
    with open("picsellia_token", "r", encoding="utf-8") as f:
        picsellia_token = f.read().strip()

    config["picsellia_token"] = picsellia_token

    # return config
    return config

import yaml


def load_config() -> dict:

    # load config from yaml file
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # load token from file
    with open("picsellia_token", "r") as f:
        picsellia_token = f.read().strip()

    config["picsellia_token"] = picsellia_token

    # return config
    return config

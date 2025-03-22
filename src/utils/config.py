import yaml


def load_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

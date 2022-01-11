from pathlib import Path

import yaml


def load_config(file=None):
    config_path = Path(file)
    try:
        with config_path.open() as config:
            try:
                return yaml.safe_load(config)
            except yaml.YAMLError as exc:
                print(exc)
    except FileNotFoundError:
        print('Defaulting...')
        return yaml.safe_load(r'config/default_configuration.yaml')

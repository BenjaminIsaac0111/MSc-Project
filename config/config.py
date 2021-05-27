from pathlib import Path
import yaml

config_path = Path('config/configuration.yaml')
with config_path.open() as config:
    try:
        CONFIG = yaml.safe_load(config)
    except yaml.YAMLError as exc:
        print(exc)

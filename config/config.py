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
        return {'CACHE': 'cache\\',
                'DATA_DIR': 'data\\',
                'TRAINING_DATA': 'models\\TrainingData.txt',
                'TEST_DATA': 'models\\TestData.txt',
                'PATCHES_DIR': 'data\\HGDL3\\',
                'XML_DIRECTORY': 'data\\MIM_XML_CONVERSIONS\\',
                'ASSOCIATED_FILE_PATTERN': 'w.*scores.xml',
                'DOWNSAMPLE_FACTOR': 2,
                'PATCH_SIZE':
                    {'WIDTH': 512,
                     'HEIGHT': 1024},
                'MASK': {'RADIUS': 32}}


CONFIG = load_config(file='config\\default_configuration.yaml')

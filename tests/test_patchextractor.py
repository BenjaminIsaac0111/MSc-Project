from multiprocessing import Pool
from functools import partial
import pytest
from Processing import PatchExtractor


@pytest.fixture
def configuration():
    configuration = 'tests\\test_configuration.yaml'
    return configuration


def test_1_extraction(configuration):
    patch_extractor = PatchExtractor(config_file=configuration)
    for file in patch_extractor.svs_files:
        patch_extractor.load_svs(file)
        patch_extractor.load_associated_file()

        try:
            patch_extractor.extract_points()
        except AttributeError as e:
            print('\tNo associated file loaded for {}. '.format(patch_extractor.svs_name))
            print('\tCheck RegEx pattern or Missing File?\n'.format())
            patch_extractor.close_svs()
            continue

        patch_extractor.extract_patches(visualise_mask=True)
        patch_extractor.close_svs()

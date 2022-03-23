import pytest
from SVSLoader.Processing.pointannotationpatchextractor import PointAnnotationPatchExtractor
from SVSLoader.Processing.dcrfmaskextractor import DenseCRFMaskExtractor


@pytest.fixture
def configuration():
    configuration = 'tests\\test_configuration.yaml'
    return configuration


def test_1_patch_extraction(configuration):
    patch_extractor = PointAnnotationPatchExtractor(config=configuration)
    for i, file in enumerate(patch_extractor.svs_files):
        patch_extractor.load_svs_by_id(file)
        patch_extractor.load_associated_file()
        patch_extractor.parse_annotation()
        patch_extractor.build_patch_filenames()
        for j, _ in enumerate(patch_extractor.patch_filenames):
            patch_extractor.read_patch_region(loc_idx=j)
            patch_extractor.build_mask()
            patch_extractor.ground_truth_mask = patch_extractor.ground_truth_mask * 127
            patch_extractor.extract_patch()
            patch_extractor.save_patch()
            if j > 10:
                break
        patch_extractor.close_svs()


def test_2_dcrf_extraction(configuration):
    dcrf_extractor = DenseCRFMaskExtractor(config_file=configuration)
    for i, file in enumerate(dcrf_extractor.svs_files):
        dcrf_extractor.load_svs_by_id(file)
        dcrf_extractor.load_associated_file()
        dcrf_extractor.parse_annotation()
        dcrf_extractor.build_patch_filenames()
        for j, _ in enumerate(dcrf_extractor.patch_filenames):
            dcrf_extractor.read_patch_region(loc_idx=j)
            dcrf_extractor.build_mask()
            dcrf_extractor.ground_truth_mask = dcrf_extractor.ground_truth_mask * 127
            dcrf_extractor.extract_patch()
            dcrf_extractor.save_patch()
            if j > 10:
                break
        dcrf_extractor.close_svs()

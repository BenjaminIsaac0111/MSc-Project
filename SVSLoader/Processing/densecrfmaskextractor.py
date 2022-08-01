import os
import numpy as np
from PIL import Image
from cv2 import resize
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from SVSLoader.Loaders.resultloader import ResultLoader
from SVSLoader.Processing.pointannotationpatchextractor import PointAnnotationPatchExtractor
from SVSLoader.Utils.utils import create_circular_mask


class DenseCRFMaskExtractor(PointAnnotationPatchExtractor):
    def __init__(self, config_file=None):
        super().__init__(config=config_file)
        self.results_loader = ResultLoader(config=self.CONFIG)
        self.tp_result_patch_filenames = [str(name) for name in self.results_loader.TRUE_POSITIVES]
        self.N_CLASSES = self.results_loader.N_CLASSES
        self.feature_embedding_resolution = self.results_loader.EMBEDDINGS.shape[1:]
        self.center = None
        self.enforce_gt_cir_mask = create_circular_mask(
            h=self.loaded_rgb_patch_img.shape[0],
            w=self.loaded_rgb_patch_img.shape[1],
            radius=gt_assertion_mask_radius
        )

    def build_ground_truth_mask(self, gaussian_sxy=4, bilateral_sxy=48, crf_n_iter=4,
                                use_cir_mask=True, gt_assertion_mask_radius=4):

        patch_filename = self.patch_filenames[self.point_index]
        model_softmax_output = self.results_loader.get_prediction_by_patch_name(patch_filename)
        patch_centroid_truth = self.results_loader.get_centroid_truth_by_patch_name(patch_filename)
        cir_mask = create_circular_mask(
            h=self.loaded_rgb_patch_img.shape[0],
            w=self.loaded_rgb_patch_img.shape[1],
            radius=self.CONFIG['CONTEXT_MASK_RADIUS']
        )

        if self.center is None:
            self.center = (
                int(self.loaded_rgb_patch_img.shape[1] / 2),
                int(self.loaded_rgb_patch_img.shape[0] / 2)
            )

        model_softmax_output = resize(model_softmax_output, dsize=self.patch_w_h)
        im_softmax = model_softmax_output
        # Assert that the ground truth is preserved when applying the CRF.
        gt_assert = np.zeros(shape=self.N_CLASSES)
        gt_assert[patch_centroid_truth] = 254
        im_softmax[self.enforce_gt_cir_mask] = gt_assert

        feat_first = im_softmax.transpose((2, 0, 1)).reshape((self.N_CLASSES, -1))
        unary = unary_from_softmax(feat_first)
        unary = np.ascontiguousarray(unary)  # As C array.

        d = dcrf.DenseCRF2D(
            self.loaded_rgb_patch_img.shape[1],
            self.loaded_rgb_patch_img.shape[0],
            self.N_CLASSES
        )
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(
            sxy=gaussian_sxy,
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )
        d.addPairwiseBilateral(
            sxy=bilateral_sxy, srgb=13, rgbim=self.loaded_rgb_patch_img,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC
        )

        q = d.inference(crf_n_iter)

        res = np.argmax(q, axis=0).reshape(
            (self.loaded_rgb_patch_img.shape[0],
             self.loaded_rgb_patch_img.shape[1])
        )

        res[res != patch_centroid_truth] = -1  # Get rid of other classes.
        if use_cir_mask:
            res[~cir_mask] = -1  # Set all values outside the context to be ignored by the model when training.
        res[res == patch_centroid_truth] = patch_centroid_truth
        res = res + 1  # Offset for compatibility with HGDL3. i.e. 0 is void, 1 is Non-Informative.
        channel = np.zeros(shape=res.shape)

        self.ground_truth_mask = np.dstack([channel, channel, res]).astype('uint8')

    def save_patch(self):
        patch_filepath = f'{self.CONFIG["PATCHES_DIR"]}\\{self.patch_filenames[self.point_index]}'
        Image.fromarray(self.patch).save(fp=patch_filepath)

    def run_extraction(self):
        if not os.path.exists(f'{self.CONFIG["PATCHES_DIR"]}'):
            os.mkdir(f'{self.CONFIG["PATCHES_DIR"]}\\')
        existing_patches = os.listdir(path=self.CONFIG['PATCHES_DIR'])
        for i, file in enumerate(self.svs_files):
            self.load_svs_by_id(file)
            self.load_associated_file()

            try:
                self.parse_annotation()
            except AttributeError as e:
                m = f'\tNo associated file found for {file} using RegEx: {self.CONFIG["ASSOCIATED_FILE_PATTERN"]}.\n'
                self.loader_message += m
                self.print_loader_message()
                continue

            self.build_patch_filenames()

            for j, filename in enumerate(self.patch_filenames):
                if filename in existing_patches:
                    continue
                self.read_patch_region(loc_idx=j)
                self.build_ground_truth_mask()
                self.build_patch()
                self.save_patch()
                self.print_loader_message()

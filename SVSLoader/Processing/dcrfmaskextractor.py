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
        self.results_loader.generate_sample(random_sample_size=128)
        self.result_patch_filenames = list(self.results_loader.get_patch_names())
        self.N_CLASSES = self.results_loader.N_CLASSES
        self.feature_embedding_resolution = self.results_loader.EMBEDDINGS.shape[1:]

    def build_ground_truth_mask(self, gaussian_sxy=10, bilateral_sxy=50, crf_n_iter=5):
        patch_filename = self.patch_filenames[self.point_index]
        seg_mask = self.results_loader.get_prediction_by_patch_name(patch_filename)
        patch_centroid_truth = self.results_loader.get_centroid_truth_by_patch_name(patch_filename)
        cir_mask = create_circular_mask(h=self.loaded_rgb_patch_img.shape[0], w=self.loaded_rgb_patch_img.shape[1],
                                        radius=self.CONFIG['CRF_CONTEXT_MASK_RADIUS'])
        seg_mask = resize(seg_mask, dsize=self.patch_w_h)
        im_softmax = seg_mask
        feat_first = im_softmax.transpose((2, 0, 1)).reshape((self.N_CLASSES, -1))
        unary = unary_from_softmax(feat_first)
        unary = np.ascontiguousarray(unary)  # As C array.
        d = dcrf.DenseCRF2D(self.loaded_rgb_patch_img.shape[1], self.loaded_rgb_patch_img.shape[0], self.N_CLASSES)
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=gaussian_sxy, compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=bilateral_sxy, srgb=13, rgbim=self.loaded_rgb_patch_img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
        q = d.inference(crf_n_iter)
        res = np.argmax(q, axis=0).reshape((self.loaded_rgb_patch_img.shape[0], self.loaded_rgb_patch_img.shape[1]))
        res[res != patch_centroid_truth] = 0
        res[~cir_mask] = 0
        res[res == patch_centroid_truth] = patch_centroid_truth
        channel = np.zeros(shape=res.shape)
        self.ground_truth_mask = np.dstack([channel, channel, res]).astype('uint8')

    def save_patch(self):
        if not os.path.exists(f'{self.CONFIG["PATCHES_DIR"]}CRF_Masks\\'):
            os.mkdir(f'{self.CONFIG["PATCHES_DIR"]}CRF_Masks\\')
        patch_filepath = f'{self.CONFIG["PATCHES_DIR"]}CRF_Masks\\{self.patch_filenames[self.point_index]}'
        Image.fromarray(self.patch).save(fp=patch_filepath)

import h5py
from functools import lru_cache
import numpy as np
from matplotlib import pyplot as plt
from SVSLoader.Config import load_config
from SVSLoader.Utils.utils import seglabel2colourmap


class ResultLoader:
    def __init__(self, config=None):
        if type(config) == str:
            self.CONFIG = load_config(config)
        elif type(config) == dict:
            self.CONFIG = config
        self.f = h5py.File(self.CONFIG['RESULTS_DATABASE'], 'r')
        self.H5_OUTDIR = self.CONFIG['RESULTS_H5_DIR'] + self.CONFIG['MODEL_NAME'] + '/'
        self.CLASSES = self.CONFIG['CLASS_COMPONENTS']
        self.N_CLASSES = len(self.CLASSES)
        self.RESULTS_SET = self.H5_OUTDIR + list(self.f[self.H5_OUTDIR].keys())[self.CONFIG['RESULT_SET_IDX']] + '/'
        self.PATCH_NAMES = np.array(self.f[self.RESULTS_SET + 'Patch_Names'])
        self.EMBEDDINGS = self.f[self.RESULTS_SET + 'Embeddings']
        self.PREDICTIONS = self.f[self.RESULTS_SET + 'Predictions']
        self.CENTROIDS_PREDS = np.squeeze(self.PREDICTIONS[self.PREDICTIONS.attrs['Centroids']])
        self.CENTROIDS_TRUTH = self.EMBEDDINGS.attrs['Centroids_True_Class']
        self.CENTROIDS_EMBEDDINGS = np.squeeze(self.EMBEDDINGS[self.EMBEDDINGS.attrs['Centroids']])
        self.samples_idx = None
        self.random_seed = self.CONFIG['RANDOM_SEED']
        if self.random_seed:
            np.random.seed(seed=self.random_seed)
        self.generate_sample()

    def generate_sample(self, random_sample_size=None):
        if not random_sample_size:
            self.samples_idx = list(range(len(self.PATCH_NAMES)))
        else:
            idx = np.random.choice(len(self.PATCH_NAMES), size=random_sample_size, replace=False)
            self.samples_idx = sorted(idx)

    @lru_cache
    def get_patch_sample_names(self):
        return self.PATCH_NAMES[self.samples_idx]

    def get_patch_names(self):
        return self.PATCH_NAMES

    @lru_cache
    def get_embedding_samples(self):
        return self.EMBEDDINGS[self.samples_idx]

    @lru_cache
    def get_prediction_samples(self):
        return self.PREDICTIONS[self.samples_idx]

    def get_prediction_by_patch_name(self, patch_name=None):
        return np.squeeze(self.PREDICTIONS[np.where(self.PATCH_NAMES == patch_name)])

    @lru_cache
    def get_centroid_embeddings_samples(self):
        return self.CENTROIDS_EMBEDDINGS[self.samples_idx]

    @lru_cache
    def get_segmentation_maps_samples(self):
        return np.argmax(self.PREDICTIONS[self.samples_idx], axis=3)

    @lru_cache
    def get_segmentation_maps(self):
        return np.argmax(self.PREDICTIONS, axis=3)

    @lru_cache
    def get_rgb_segmentation_maps_sample(self, cmap=plt.cm.tab10.colors):
        return seglabel2colourmap(self.get_segmentation_maps_samples(), cmap_lut=cmap)

    def get_rgb_segmentation_maps(self, cmap=plt.cm.tab10.colors):
        return seglabel2colourmap(self.get_segmentation_maps(), cmap_lut=cmap)

    @lru_cache
    def get_true_positives_preds(self):
        return np.equal(self.CENTROIDS_PREDS, np.argmax(self.CENTROIDS_TRUTH, axis=1))

    def get_centroid_embeddings(self):
        return self.CENTROIDS_EMBEDDINGS

    @lru_cache
    def get_centroid_truth_sample(self):
        return self.CENTROIDS_TRUTH[self.samples_idx]

    def get_centroid_truth_by_patch_name(self, patch_name=None):
        return np.squeeze(self.CENTROIDS_TRUTH[np.where(self.PATCH_NAMES == patch_name)])

    @staticmethod
    def set_random_seed(random_seed=None):
        np.random.seed(seed=random_seed)

    def set_predefind_samples(self, samples_idx):
        self.samples_idx = samples_idx

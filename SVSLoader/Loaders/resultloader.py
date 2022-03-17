import h5py
from functools import lru_cache
import numpy as np
from SVSLoader.Config import load_config


class ResultLoader:
    def __init__(self, config_path=None):
        self.CONFIG = load_config(config_path)
        self.f = h5py.File(self.CONFIG['RESULTS_DATABASE'], 'r')
        self.H5_OUTDIR = self.CONFIG['RESULTS_H5_DIR'] + self.CONFIG['MODEL_NAME'] + '/'
        self.CLASSES = self.CONFIG['CLASS_COMPONENTS']
        self.N_CLASSES = len(self.CLASSES)
        self.RESULTS_SET = self.H5_OUTDIR + list(self.f[self.H5_OUTDIR].keys())[self.CONFIG['RESULT_SET_IDX']] + '/'
        self.PATCH_NAMES = self.f[self.RESULTS_SET + 'Patch_Names']
        self.EMBEDDINGS = self.f[self.RESULTS_SET + 'Embeddings']
        self.PREDICTIONS = self.f[self.RESULTS_SET + 'Predictions']
        self.CENTROIDS_PREDS = np.squeeze(self.PREDICTIONS[self.PREDICTIONS.attrs['Centroids']])
        self.CENTROIDS_TRUTH = self.EMBEDDINGS.attrs['Centroids_True_Class']
        self.CENTROIDS_EMBEDDINGS = np.squeeze(self.EMBEDDINGS[self.EMBEDDINGS.attrs['Centroids']])
        self.samples_idx = None
        self.generate_sample()

    def generate_sample(self, random_sample_size=None):
        if not random_sample_size:
            self.samples_idx = list(range(len(self.PATCH_NAMES)))
        else:
            idx = np.random.choice(len(self.PATCH_NAMES), size=random_sample_size, replace=False)
            self.samples_idx = sorted(idx)

    @lru_cache
    def get_patch_names(self):
        return self.PATCH_NAMES[self.samples_idx]

    @lru_cache
    def get_embedding_samples(self):
        return self.EMBEDDINGS[self.samples_idx]

    @lru_cache
    def get_prediction_samples(self):
        return self.PREDICTIONS[self.samples_idx]

    @lru_cache
    def get_centroid_embeddings_samples(self):
        return self.CENTROIDS_EMBEDDINGS[self.samples_idx]

    @lru_cache
    def get_segmentation_maps(self):
        return np.argmax(self.PREDICTIONS[self.samples_idx], axis=3)

    @lru_cache
    def get_true_positives_preds(self):
        return np.equal(self.CENTROIDS_PREDS, np.argmax(self.CENTROIDS_TRUTH, axis=1))

    def get_centroid_embeddings(self):
        return self.CENTROIDS_EMBEDDINGS

    def set_predefind_samples(self, samples_idx):
        self.samples_idx = samples_idx

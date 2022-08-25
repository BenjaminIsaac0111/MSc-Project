import h5py
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
from SVSLoader.Config import load_config
from SVSLoader.Utils.utils import seglabel2colourmap


class ResultLoader:
    def __init__(self, config=None, random_seed=None):
        if type(config) == str:
            self.CONFIG = load_config(config)
        elif type(config) == dict:
            self.CONFIG = config
        self.f = h5py.File(self.CONFIG['RESULTS_DATABASE'], 'r')
        self.H5_OUTDIR = self.CONFIG['RESULTS_H5_DIR'] + self.CONFIG['MODEL_NAME'] + '/'
        self.CLASSES = self.CONFIG['CLASS_COMPONENTS']  # TODO should store in the .h5 itself.
        self.N_CLASSES = len(self.CLASSES)
        self.RESULTS_SETS = self.H5_OUTDIR + list(self.f[self.H5_OUTDIR].keys())[self.CONFIG['RESULT_SET_IDX']] + '/'
        self.PATCH_NAMES = np.array(self.f[self.RESULTS_SETS + 'Patch_Names']).astype(str)
        self.EMBEDDINGS = self.f[self.RESULTS_SETS + 'Embeddings']
        self.EMBEDDINGS_SHAPE = self.EMBEDDINGS.shape[1:]
        self.PREDICTIONS = self.f[self.RESULTS_SETS + 'Predictions']
        self.PREDICTIONS_SHAPE = self.PREDICTIONS.shape[1:]
        self.CENTROIDS_PREDS = np.squeeze(self.PREDICTIONS[self.PREDICTIONS.attrs['Centroids']])
        self.CENTROIDS_TRUTH = self.EMBEDDINGS.attrs['Centroids_True_Class'][:]
        self.CENTROIDS_EMBEDDINGS = np.squeeze(self.EMBEDDINGS[self.EMBEDDINGS.attrs['Centroids']])
        self.CORRECTNESS_MASK = np.equal(np.argmax(self.CENTROIDS_PREDS, axis=1), self.CENTROIDS_TRUTH)
        self.samples_idx = None
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(seed=self.random_seed)
        self.build_patch_sample()

    @staticmethod
    def set_random_seed(random_seed=None):
        np.random.seed(seed=random_seed)

    def build_patch_sample(self, random_sample_size=None):
        if not random_sample_size:
            self.samples_idx = list(range(len(self.PATCH_NAMES)))
        else:
            idx = np.random.choice(len(self.PATCH_NAMES), size=random_sample_size, replace=False)
            self.samples_idx = sorted(idx)

    @lru_cache
    def get_patch_sample_names(self):
        return self.PATCH_NAMES[self.samples_idx]

    @lru_cache
    def get_centroid_preds_sample(self):
        return np.argmax(self.CENTROIDS_PREDS[self.samples_idx], axis=1)

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
    def get_segmentation_map_samples(self):
        return np.argmax(self.PREDICTIONS[self.samples_idx], axis=3)

    @lru_cache
    def get_segmentation_maps(self):
        return np.argmax(self.PREDICTIONS, axis=3)

    @lru_cache
    def get_rgb_segmentation_map_samples(self, cmap=plt.cm.tab10.colors):
        return seglabel2colourmap(self.get_segmentation_map_samples(), cmap_lut=cmap)

    @lru_cache
    def get_rgb_segmentation_maps(self, cmap=plt.cm.tab10.colors):
        return seglabel2colourmap(self.get_segmentation_maps(), cmap_lut=cmap)

    @lru_cache
    def get_true_positives_preds(self):
        return np.equal(self.CENTROIDS_PREDS, np.argmax(self.CENTROIDS_TRUTH, axis=1))

    @lru_cache
    def get_centroid_truth_samples(self):
        return self.CENTROIDS_TRUTH[self.samples_idx]

    def get_centroid_truth_by_patch_name(self, patch_name=None):
        return np.squeeze(self.CENTROIDS_TRUTH[np.where(self.PATCH_NAMES == patch_name)])

    def get_prediction_by_patch_name(self, patch_name=None):
        return np.squeeze(self.PREDICTIONS[np.where(self.PATCH_NAMES == patch_name)])

    @lru_cache
    def get_nn_lookup(self, **kwargs):
        n_neighbors_lookup = NearestNeighbors(**kwargs)
        n_neighbors_lookup = n_neighbors_lookup.fit(X=self.CENTROIDS_EMBEDDINGS)
        return n_neighbors_lookup

    def plot_dim_reduction(self, n_components=3, figsize=(10.80, 10.80), mask=None):
        if mask is None:
            mask = np.ones(len(self.CENTROIDS_TRUTH)).astype(bool)
        dim_reduct = PCA(n_components=n_components).fit(X=self.CENTROIDS_EMBEDDINGS.T)
        components_df = pd.DataFrame(dim_reduct.components_.T)
        components_df['Y'] = self.CENTROIDS_TRUTH
        plt.figure(figsize=figsize)
        sns.pairplot(components_df[mask],
                     hue='Y',
                     palette=sns.color_palette("tab10", len(set(components_df['Y'][mask]))))
        plt.show()

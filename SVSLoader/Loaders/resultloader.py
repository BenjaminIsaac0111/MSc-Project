import os

import h5py
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
from SVSLoader.Config import load_config
from SVSLoader.Utils.utils import seglabel2colourmap
from PIL import Image


class ResultLoader:
    def __init__(self, config=None, random_seed=None):
        if type(config) == str:
            self.CONFIG = load_config(config)
        elif type(config) == dict:
            self.CONFIG = config
        self.f = h5py.File(self.CONFIG['RESULTS_DATABASE'], 'r')
        self.H5_OUTDIR = 'outputs/' + self.CONFIG['MODEL_NAME'] + '/'
        self.RESULTS_SETS = self.H5_OUTDIR + list(self.f[self.H5_OUTDIR].keys())[self.CONFIG['RESULT_SET_IDX']] + '/'
        self.PATCH_NAMES = np.array(self.f[self.RESULTS_SETS + 'Patch_Names']).astype(str)
        self.EMBEDDINGS = self.f[self.RESULTS_SETS + 'Embeddings']
        self.PREDICTIONS = self.f[self.RESULTS_SETS + 'Predictions']
        self.CENTROIDS_PREDS = np.squeeze(self.PREDICTIONS[self.PREDICTIONS.attrs['Centroids']])
        self.CENTROIDS_TRUTH = self.EMBEDDINGS.attrs['Centroids_True_Class'][:]
        self.CENTROIDS_EMBEDDINGS = np.squeeze(self.EMBEDDINGS[self.EMBEDDINGS.attrs['Centroids']])
        self.CORRECTNESS_MASK = np.equal(np.argmax(self.CENTROIDS_PREDS, axis=1), self.CENTROIDS_TRUTH)
        self.CLASS_COMPONENTS = self.PREDICTIONS.attrs['Components']
        self.N_CLASSES = len(self.CLASS_COMPONENTS)
        self.samples_idx = None
        self.random_seed = random_seed
        if self.random_seed:
            np.random.seed(seed=self.random_seed)
        self.build_patch_sample()
        self.gt_confidence_samples = None
        self.CONFIDENCE_MODEL = None
        self.NN_LOOKUP = None

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

    @lru_cache
    def get_centroid_truth_by_patch_name(self, patch_name=None):
        return np.squeeze(self.CENTROIDS_TRUTH[np.where(self.PATCH_NAMES == patch_name)])

    @lru_cache
    def get_prediction_by_patch_name(self, patch_name=None):
        return np.squeeze(self.PREDICTIONS[np.where(self.PATCH_NAMES == patch_name)])

    @lru_cache
    def load_patch(self, idx):
        file = os.path.split(self.PATCH_NAMES[idx])[-1]
        filepath = f"{self.CONFIG['PATCHES_DIR']}{file}"
        patch_img = np.array(Image.open(filepath))
        xs = patch_img.shape[0] // 2
        return patch_img[:, 0:xs], patch_img[:, xs:]

    @lru_cache
    def get_nn_lookup(self, **kwargs):
        n_neighbors_lookup = NearestNeighbors(**kwargs)
        n_neighbors_lookup = n_neighbors_lookup.fit(X=self.CENTROIDS_EMBEDDINGS)
        return n_neighbors_lookup

    def build_confidence_model(self, **kwargs):
        self.NN_LOOKUP = self.get_nn_lookup(**kwargs)
        nn_distances, nn_ids = self.NN_LOOKUP.kneighbors(self.CENTROIDS_EMBEDDINGS)
        # Computes local correctness of n nearest neighbours.
        correctness_values = self.calculate_local_correctness(nn_ids)
        # Compute the mean distance to the n nearest neighbours.
        mean_nn_distance = self.calculate_mean_nn_distances(nn_distances)
        # Compute the local entropy of the n nearest neighbours.
        entropy_values = self.calculate_local_entropy(nn_ids)

        df = pd.DataFrame(
            np.array(
                [correctness_values,
                 mean_nn_distance,
                 entropy_values]
            ).T
        )
        df.columns = ['local_correctness', 'mean_nn_distance', 'local_entropy']

        self.gt_confidence_samples = df
        self.regression_analysis()
        df['prob_incorrect'], df['prob_correct'] = self.CONFIDENCE_MODEL.predict_proba(df).T

    def build_pseudo_labels(self, **kwargs):
        for idx, _ in self.samples_idx:
            patch_img = self.load_patch(idx)

            # Get confidence values #
            nn_distances, nn_ids = self.NN_LOOKUP.kneighbors(self.EMBEDDINGS[idx])
            # Computes local correctness of n nearest neighbours.
            correctness_values = self.calculate_local_correctness(nn_ids)
            # Compute the mean distance to the n nearest neighbours.
            mean_nn_distance = self.calculate_mean_nn_distances(nn_distances)
            # Compute the local entropy of the n nearest neighbours.
            entropy_values = self.calculate_local_entropy(nn_ids)

            sample_confidence_values = np.array(
                [
                    correctness_values,
                    mean_nn_distance,
                    entropy_values
                ]
            ).T

            prob_incorrect, prob_correct = self.CONFIDENCE_MODEL.predict_proba(sample_confidence_values).T
            sample_confidence_values.append(prob_incorrect, axis=1)
            sample_confidence_values.append(prob_correct, axis=1)

    def calculate_local_entropy(self, nn_ids):
        labelled_nn = self.CENTROIDS_TRUTH[nn_ids]
        local_nn_probs = [
            np.unique(labelled, return_counts=True)[1] / len(labelled) for labelled in labelled_nn
        ]
        entropy_values = np.array([entropy(pk) for pk in local_nn_probs])
        # Normalise entropy values and calculate the maximum possible entropy value.
        unique_classes = list(self.CLASS_COMPONENTS.keys())
        # H_max == log(c) where c is n unique classes.
        H_max = entropy(np.ones(len(unique_classes)) / len(unique_classes))
        # Normalised entropy within range [0, 1].
        entropy_values = (entropy_values / H_max)
        return entropy_values

    def calculate_mean_nn_distances(self, nn_distances):
        feature_space_origin = np.ones(shape=self.get_embedding_samples().shape[-1])
        D_max = distance.minkowski(feature_space_origin - 1, feature_space_origin * 255)
        mean_nn_distance = np.mean(nn_distances, axis=1)
        mean_nn_distance = (mean_nn_distance / D_max)
        return mean_nn_distance

    def calculate_local_correctness(self, nn_ids):
        correctness_values = (np.sum(self.CORRECTNESS_MASK[nn_ids].astype(int), axis=1) / nn_ids.shape[1])
        return correctness_values

    def regression_analysis(self, verbose=True, class_weight='unbalanced'):
        clf = LogisticRegression(random_state=7, class_weight=class_weight)
        self.CONFIDENCE_MODEL = clf.fit(self.gt_confidence_samples, self.CORRECTNESS_MASK)
        clf_score = clf.score(self.gt_confidence_samples, self.CORRECTNESS_MASK)
        if verbose:
            print(f'Bal_Score: {clf_score}')
            print(f'Intercept: {clf.intercept_}')
            print(f'Coefficients: {clf.feature_names_in_}{clf.coef_[0]}\n')

    def plot_dim_reduction(self, n_components=2, figsize=(10.80, 10.80), mask=None):
        if mask is None:
            mask = np.ones(len(self.CENTROIDS_TRUTH)).astype(bool)
        dim_reduction = PCA(n_components=n_components).fit(X=self.CENTROIDS_EMBEDDINGS.T)
        components_df = pd.DataFrame(dim_reduction.components_.T)
        components_df['Y'] = self.CENTROIDS_TRUTH
        plt.figure(figsize=figsize)
        sns.pairplot(components_df[mask],
                     hue='Y',
                     palette=sns.color_palette("tab10", len(set(components_df['Y'][mask]))))
        plt.show()

    def plot_pca(self, n_components=2, figsize=(10.80, 10.80), mask=None):
        if mask is None:
            mask = np.ones(len(self.CENTROIDS_TRUTH)).astype(bool)
        dim_reduction = PCA(n_components=n_components).fit(X=self.CENTROIDS_EMBEDDINGS.T)
        components_df = pd.DataFrame(dim_reduction.components_.T)
        components_df['Y'] = self.CENTROIDS_TRUTH
        plt.figure(figsize=figsize)
        sns.scatterplot(components_df[mask],
                        hue='Y',
                        palette=sns.color_palette("tab10", len(set(components_df['Y'][mask]))))
        plt.show()

import os

import numpy as np
from sklearn.utils import class_weight

from src import utils
import random
from sklearn.model_selection import KFold


def get_classes_from_data_dir(directory=None):
    """
    This will fetch a list of the classes from extracted patches in a specified directory.
    :param directory: The directory containing the extracted patches.
    :return: a list of the class for each patch.
    """
    return [[file, str(int(file[-5]) + 1)] for file in os.listdir(directory) if file.endswith('.png')]


def train_test_split_sample(split=0.5, directory=None):
    """
    Build the train test split by reading the svs ids in a given directory containing extracted patches.
    Needs the patch filenames in a directory to follow the <id>_<patch_no>_<class>.png format. (May change)
    :param split percentage
    :param directory: directory containing the split
    :return training sample list of svs images ids and test sample list of svs image ids
    """
    training_list = [file + '\t' + cla for file, cla in get_classes_from_data_dir(directory)]
    ids = [image_id[:6] for image_id in training_list]
    sample_image_ids = random.sample(set(ids), round(len(set(ids)) * split))
    training_sample = [patch for patch in training_list if patch[:6] in sample_image_ids]
    testing_sample = [patch for patch in training_list if patch[:6] not in sample_image_ids]
    return training_sample, testing_sample


def k_fold_cross_validation_from_directory(n_splits=5, directory=None, random_state=None):
    files = [file + '\t' + cla for file, cla in get_classes_from_data_dir(directory)]
    svs_ids = set([svs_id.split('_')[0] for svs_id in files])
    if n_splits > 1:
        kf = KFold(n_splits=n_splits, random_state=random_state)
        return kf.split(svs_ids), np.array(files)
    else:
        return svs_ids, np.array(files)


def build_train_test_lists(destination_dir=None, n_splits=5):
    data_fold_indices, files = k_fold_cross_validation_from_directory(n_splits=n_splits, directory=destination_dir)
    i = 0
    for x_training_set, y_test_set in data_fold_indices:
        training_patches_by_svs_id = [file for file in files if file.split('_')[0] in x_training_set]
        test_patches_by_svs_id = [file for file in files if file.split('_')[0] in y_test_set]
        utils.write_list(training_patches_by_svs_id, destination_dir + '\\fold_' + str(i) + '_TrainingData.txt')
        utils.write_list(test_patches_by_svs_id[y_test_set], destination_dir + '\\fold_' + str(i) + '_TestData.txt')
        i += 1


def class_weights(directory=None):
    """
    Estimate class weights for unbalanced datasets.
    :param directory directory containing patches with class in filenames.
    :return class weights.
    """
    classes = get_classes_from_data_dir(directory)
    classes = [cls for _, cls in classes]
    return class_weight.compute_class_weight('balanced',
                                             np.unique(classes),
                                             classes)


# build_train_test_lists(destination_dir=r'E:\32pxToyData', n_splits=5)

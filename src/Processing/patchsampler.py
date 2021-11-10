import os
import re
import numpy as np
from sklearn.utils import class_weight
import random
from sklearn.model_selection import KFold


def get_patch_meta_data_from_dir(directory=None):
    """
    [<filename>[0], [<institute_id>[0]_<patient_id>[1]_<svs_id>[2]_<patch_no>[3]_<class>[4].png[5]]]
    """
    return [[file, re.split('[_.]', file)] for file in os.listdir(directory) if file.endswith('.png')]


# TODO: Kinda redundant but need to modify its dependencies to work with newer train_test_split_by_meta_id().
def get_classes_from_data_dir(directory=None):
    """
    This will fetch a list of the classes from extracted patches in a specified directory.
    :param directory: The directory containing the extracted patches.
    :return: a list of the class for each patch.
    """
    return [[file, str(int(file[-5]) + 1)] for file in os.listdir(directory) if file.endswith('.png')]


def k_fold_cross_validation_from_directory(n_splits=5, directory=None, random_state=None):
    files = [file + '\t' + cla for file, cla in get_classes_from_data_dir(directory)]
    svs_ids = set([svs_id.split('_')[0] for svs_id in files])
    if n_splits > 1:
        kf = KFold(n_splits=n_splits, random_state=random_state)
        return kf.split(svs_ids), np.array(files)
    else:
        return svs_ids, np.array(files)


def class_weights_from_directory(directory=None):
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


def train_test_split_by_meta_id(directory=None, split=0.7, id_level=0, seed=7):
    """
    Build the train test split by reading the ids in a given directory containing extracted patches. Needs
    the patch filenames in a directory to follow the following list data structure;
    [[0]<filename>,[1][[0]<institute_id>_(if available...)[1]<patient_id>_[2]<svs_id>_[3]<patch_no>_[4]<class>[5].png]]
    :param seed: set the random seed.
    :param id_level: what id level in the filename to split the data by.
    :param split: to what ratio of split should be made.
    :param directory: directory of patch images where the patches follow the aforementioned filename convention.
    :return list of training patches and list of test patches via id level.
    """
    random.seed(seed)
    patch_list = [meta for meta in get_patch_meta_data_from_dir(directory)]
    ids = [patch[1][id_level] for patch in patch_list]
    sample_image_ids = random.sample(set(ids), round(len(set(ids)) * split))
    training_sample = [patch[0] + '\t' + patch[1][-2] for patch in patch_list if patch[1][id_level] in sample_image_ids]
    testing_sample = [patch[0] + '\t' + patch[1][-2] for patch in patch_list if patch[1][id_level] not in sample_image_ids]
    if any(item for item in training_sample if item in testing_sample):
        print('Overlapping sets!')
    return training_sample, testing_sample

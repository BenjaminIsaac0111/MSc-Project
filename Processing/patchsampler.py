from utils import utils
import random
from sklearn.model_selection import KFold


def train_test_split_sample(split=0.5, directory=None):
    """
    Build the train test split by reading the svs ids in a given directory containing extracted patches.
    Needs the patch filenames in a directory to follow the <id>_<patch_no>_<class>.png format. (May change)
    :param split percentage
    :param directory: directory containing the split
    :return training sample list of svs images ids and test sample list of svs image ids
    """
    training_list = [file + '\t' + cla for file, cla in utils.get_classes_from_data_dir(directory)]
    ids = [image_id[:6] for image_id in training_list]
    sample_image_ids = random.sample(set(ids), round(len(set(ids)) * split))
    training_sample = [patch for patch in training_list if patch[:6] in sample_image_ids]
    testing_sample = [patch for patch in training_list if patch[:6] not in sample_image_ids]
    return training_sample, testing_sample


def k_fold_cross_validation_sampling():
    raise NotImplementedError

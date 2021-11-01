import pandas as pd
import numpy as np
import argparse
import os
from sklearn.utils import class_weight


def list_files(directory=None):
    """
    Lists files in a directory.
    :param directory: the directory to list.
    :return: a list of items in the directory.
    """
    return os.listdir(directory)


def open_dir_listings(directory_listings=None):
    """
    Open a directory listing.
    :param directory_listings: path to a directory listing file.
    :return: a list object containing the contents of the directory listing file.
    """
    with open(directory_listings) as listing:
        return [line.strip() for line in listing]


def write_list(list_to_write=None, file=None):
    """
    Write a list array to file on disk.
    :param list_to_write: a list object to write to file.
    :param file: the filename to write to.
    """
    list_to_write = map(lambda x: x + '\n', list_to_write)
    dir_listing_file = open(file=file, mode='w')
    dir_listing_file.writelines(list_to_write)
    dir_listing_file.close()


# TODO Could move into a processing/triallister.py at some point.
def find_listings_from_trial(trial_file=None, directory_listing=None):
    """
    This function will look for all the listed data that correspond with a trail excel file. A bit of a domain specific
    function.
    :param trial_file: The trial file used to build the list of svs files for the trial.
    :param directory_listing: a directory listing where the svs files are.
    :return: the found svs ids and unresolved missing ids.
    """
    trial_listing = pd.read_excel(io=trial_file, index_col=0, header=None).transpose()
    trial_listing[['Trial number', 'svs']] = trial_listing[['Trial number', 'svs']].astype('int').astype('str')

    # Resolve trial numbers for numbers of len <4 for example 401 -> 0401. Also have a bash script for this.
    trial_listing.loc[(trial_listing['Trial number'].str.len() < 4),
                      'Trial number'] = '0' + trial_listing.loc[(trial_listing['Trial number'].str.len() < 4),
                                                                'Trial number']

    trial_listing_locations = 'Y:\\' + trial_listing['Trial number'].astype('str')
    trial_listing_locations += trial_listing['Trial number'].astype('str')
    trial_listing_locations += trial_listing['svs'].astype('str')
    trial_listing_locations += '.svs'
    trial_listing_locations = list(trial_listing_locations)

    found_svs_listings = [path for path in trial_listing_locations if path in directory_listing]
    unresolved = [path for path in trial_listing_locations if path not in directory_listing]
    print('{} svs images found \n'.format(len(found_svs_listings)) +
          '{} unresolved svs images:'.format(len(unresolved)))
    return found_svs_listings, unresolved


def list_to_blocks(lst, n):
    """
    Split a list into blocks. Can be useful for pooling data for tasks.
    :param lst: the list split into blocks.
    :param n: number of blocks to split into.
    :return: the list of built blocks.
    """
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def str2bool(v):
    """
    A str to bool utility. Useful for parsing cli arguments.
    :param v: either a string or int that represents a boolean value.
    :return: a True or False python value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_classes_from_data_dir(directory=None):
    """
    This will fetch the class of extracted patches from a specified directory.
    :param directory: The directory containing the extracted patches.
    :return: a list of the class for each patch.
    """
    return [[file, str(int(file[-5]) + 1)] for file in os.listdir(directory) if file.endswith('.png')]


def get_svs_names(file=None):
    test_list = open(file=file)
    test_list = test_list.readlines()
    return list(set([id.partition('_')[0] + '.svs' for id in test_list]))


def build_training_list(directory=None):
    write_list([file + '\t' + cla for file, cla in get_classes_from_data_dir(directory)],
               directory + '\\TrainingData.txt')


def get_inv_class_proportions(counter=None):
    """
    Gets the inverse proportions of each class.
    :param counter a Counter object
    :return: unsorted and sorted dictionary of the calculated inverse proportions.
    """
    sum_of_classes = sum(counter.values())
    for key in counter:
        counter[key] = 1 - (counter[key] / sum_of_classes)
    sorted_inv_proportions = dict(sorted(counter.items()))
    return counter, sorted_inv_proportions


def class_weights(file=None):
    """
    Estimate class weights for unbalanced datasets.
    :param file file containing a list of class label for each data point in a set.
    :return class weights.
    """
    classes = open_dir_listings(file)
    classes = [int(cls[-1]) for cls in classes]
    return class_weight.compute_class_weight('balanced',
                                             np.unique(classes),
                                             classes)


def normalise(arr, t_min, t_max):
    """
    Explicit function to normalise array between given min max.
    :param arr array to normalise.
    :param t_min min value in range.
    :param t_max min value in range.
    :return normalised array.
    """
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def merge_classes(classes=None, data=None, target=None):
    """
    Merges classes in a dataset.
    :param classes classes to merge.
    :param data data points to operate on.
    :param target target class to merge into.
    :return data with merged classes
    """
    for i, d in enumerate(data):
        if d in classes:
            data[i] = target
    return data

import argparse
import os
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from matplotlib import pyplot as plt
from skimage.color import label2rgb


def open_dir_listings(directory_listings=None):
    """
    Open a directory listing.
    :param directory_listings: path to a directory listing file.
    :return: a list object containing the contents of the directory listing file.
    """
    with open(directory_listings) as listing:
        return [line.strip() for line in listing]


def write_list(list_to_write=None, file_to_write=None):
    """
    Write a list array to file on disk.
    :param list_to_write: a list object to write to file.
    :param file_to_write: the filename to write to.
    """
    list_to_write = map(lambda x: x + '\n', list_to_write)
    dir_listing_file = open(file=file_to_write, mode='w')
    dir_listing_file.writelines(list_to_write)
    dir_listing_file.close()


def list_to_blocks(lst=None, n_blocks=1):
    """
    Pure python function to split a list into blocks. Can be useful for pooling data for tasks.
    :param lst: the list split into blocks.
    :param n_blocks: number of blocks to split into.
    :return: the list of built blocks.
    """
    return [lst[i:i + n_blocks] for i in range(0, len(lst), n_blocks)]


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


def get_svs_names(file=None):
    test_list = open(file=file)
    test_list = test_list.readlines()
    return list(set([id.partition('_')[0] + '.svs' for id in test_list]))


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


def get_patch_meta_data_from_dir(directory=None):
    """
    [<filename>[0], [<institute_id>[0]_<patient_id>[1]_<svs_id>[2]_<patch_no>[3]_<class>[4].png[5]]]
    """
    return [[file, re.split('[_.]', file)] for file in os.listdir(directory) if file.endswith('.png')]


def get_patch_meta_data_from_txt(txt_file=None):
    """
    [<filename>[0], [<institute_id>[0]_<patient_id>[1]_<svs_id>[2]_<patch_no>[3]_<class>[4].png[5]]]
    """
    file_list = [file[:-2] for file in open_dir_listings(txt_file)]
    return [[file, re.split('[_.]', file)] for file in file_list if file.endswith('.png')]


def get_embedding_centers(y):
    return y[:, round(y.shape[1] / 2), round(y.shape[2] / 2), :]


def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


@lru_cache
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def seglabel2colourmap(seg_labels=None, cmap_lut=plt.cm.tab10.colors):
    return label2rgb(label=seg_labels, colors=list(cmap_lut), bg_label=-1)


def split_segmentation_classes(seg_maps=None, y_targets=None):
    return np.array([seg_maps[i] == y_targets[i] for i in range(len(seg_maps))])

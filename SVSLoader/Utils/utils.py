import argparse
import os
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from PIL import Image


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


# TODO Could move into a processing/triallister.py at some point.
def find_listings_from_trial(trial_file=None, directory_listing=None):
    """
    This function will look for all the listed data that correspond with a trial excel file.
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


def load_patch_metadata(directory=None):
    patch_listing = [patch_name for patch_name in os.listdir(directory) if patch_name.endswith(".png")]
    df = pd.DataFrame()
    df['patch_name'] = patch_listing
    cols = ['Institute_id', 'Patient_no', 'Image_id', 'Truth']
    df[cols] = df['patch_name'].str.split('_', expand=True).drop(columns=[3, 4])
    df['Patient_id'] = df['Institute_id'] + df['Patient_no']
    df['Truth'] = df['Truth'].str[0].astype(int)
    return df


def arrays_equal(a, b):
    if a.shape != b.shape:
        return False
    for ai, bi in zip(a.flat, b.flat):
        if ai != bi:
            return False
    return True


@lru_cache
def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the rbg_image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and rbg_image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def seglabel2colourmap(seg_labels=None, cmap_lut=plt.cm.tab10.colors):
    return label2rgb(label=seg_labels, colors=list(cmap_lut), bg_label=-1)


def split_segmentation_classes(seg_maps=None, y_targets=None):
    return np.array([seg_maps[i] == y_targets[i] for i in range(len(seg_maps))])


def load_patch(filepath=None):
    patch_img = np.array(Image.open(filepath))
    xs = patch_img.shape[0] // 2
    return patch_img[:, 0:xs], patch_img[:, xs:]

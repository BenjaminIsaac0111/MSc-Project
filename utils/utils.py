import pandas as pd
from config import CONFIG
import argparse


# Open a directory listing for svs images.
def open_dir_listings(directory_listings=None):
    with open(directory_listings) as listing:
        return [line.strip() for line in listing]


# Write list to disk.
def write_list(list_to_write=None, file=None):
    list_to_write = map(lambda x: x + '\n', list_to_write)
    dir_listing_file = open(file=file, mode='w')
    dir_listing_file.writelines(list_to_write)
    dir_listing_file.close()


# Find all svs files that are in a trail_file.
def find_listings(trial_file=None):
    trial_listing = pd.read_excel(io=trial_file, index_col=0, header=None).transpose()
    trial_listing[['Trial number', 'svs']] = trial_listing[['Trial number', 'svs']].astype('int').astype('str')

    # Resolve trial numbers for numbers of len <4 for example 401 -> 0401. Also have a bash script for this.
    trial_listing.loc[(trial_listing['Trial number'].str.len() < 4),
                      'Trial number'] = '0' + trial_listing.loc[
        (trial_listing['Trial number'].str.len() < 4), 'Trial number']

    trial_listing_locations = 'Y:\\' + trial_listing['Trial number'].astype('str') + \
                              '\\' + trial_listing['svs'].astype('str') + '.svs'
    trial_listing_locations = list(trial_listing_locations)
    directory_listing = open_dir_listings()
    found_svs_listings = [path for path in trial_listing_locations if path in directory_listing]
    unresolved = [path for path in trial_listing_locations if path not in directory_listing]
    print('{} svs images found \n'.format(len(found_svs_listings)) +
          '{} unresolved svs images:'.format(len(unresolved)))
    return found_svs_listings, unresolved


def split_list(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# svs_listings, unresolved_listings = find_listings(trial_file='data/CR07 TCD.xlsx')
# write_svs_listings(svs_listings=svs_listings, file='data/svs_listings.txt')

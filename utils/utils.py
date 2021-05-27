import pandas as pd
from config.config import CONFIG

# Open a directory Listing for svs image server database
def open_directory_listings(directory_listings=CONFIG['DIRECTORY_LISTING']):
    with open(directory_listings) as listing:
        return [line.strip() for line in listing]

# Write svs listing to disk.
def write_svs_listings(svs_listings=None, file=None):
    svs_listing_file = open(file=file, mode='w')
    svs_listings = map(lambda x: x + '\n', svs_listings)
    svs_listing_file.writelines(svs_listings)
    svs_listing_file.close()

# Find all svs files that are on in a trail_file.
def find_listings(trial_file=None):
    trial_listing = pd.read_excel(io=trial_file, index_col=0, header=None).transpose()
    trial_listing[['Trial number', 'svs']] = trial_listing[['Trial number', 'svs']].astype('int').astype('str')

    # Resolve trial numbers for numbers of len <4 for example 401 -> 0401.
    trial_listing.loc[(trial_listing['Trial number'].str.len() < 4),
                      'Trial number'] = '0' + trial_listing.loc[
        (trial_listing['Trial number'].str.len() < 4), 'Trial number']

    trial_listing_locations = 'Y:\\' + trial_listing['Trial number'].astype('str') + \
                              '\\' + trial_listing['svs'].astype('str') + '.svs'
    trial_listing_locations = list(trial_listing_locations)
    directory_listing = open_directory_listings()
    found_svs_listings = [path for path in trial_listing_locations if path in directory_listing]
    unresolved = [path for path in trial_listing_locations if path not in directory_listing]
    print('{} svs images found \n'.format(len(found_svs_listings)) +
          '{} unresolved svs images:'.format(len(unresolved)))
    return found_svs_listings, unresolved

# svs_listings, unresolved_listings = find_listings(trial_file='data/CR07 TCD.xlsx')
# write_svs_listings(svs_listings=svs_listings, file='data/svs_listings.txt')


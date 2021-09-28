import os
from multiprocessing import Pool
import argparse
from functools import partial
from utils import utils
from Processing import PatchExtractor
from pathlib import Path


def main(specified_svs_files=None, args=None):
    if args.config:
        patch_extractor = PatchExtractor(config_file=args.config)
    else:
        patch_extractor = PatchExtractor()

    if specified_svs_files:
        patch_extractor.svs_files = specified_svs_files

    for file in patch_extractor.svs_files:
        patch_extractor.load_svs(file)
        patch_extractor.load_associated_file()
        patch_extractor.extract_points()
        patch_extractor.extract_patches(dry=args.dry)
        patch_extractor.close_svs()


def main_pooled(specified_svs_files=None, args=None):
    extractor = PatchExtractor(config_file=args.config)
    if specified_svs_files:
        extractor.svs_files = specified_svs_files
    svs_files = utils.list_to_blocks(lst=extractor.svs_files, n=round(len(extractor.svs_files) / args.num_workers))
    pool = Pool(args.num_workers)
    main_pool = partial(main, args=args)
    pool.map_async(main_pool, list(svs_files))
    pool.close()
    pool.join()

    # training_list = [file for file in os.listdir('data/patches') if file.endswith('.png')]
    # training_list = [file + '\t' + file[-5] for file in training_list]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the whole slide images (WSL) patch extractor")

    parser.add_argument("-c", "--config", type=Path,
                        help=r'The extractor config file (YAML). Will use the default_configuration.yaml if none '
                             r'supplied.',
                        default='config\\default_configuration.yaml')

    parser.add_argument('-a', '--assoc-file-pattern', type=str,
                        help=r'Regular Expression to load the annotation file(s)',
                        default=None)

    parser.add_argument('-s', '--svs-listing', type=str,
                        help=r'Extract from svs in this svs file listing.',
                        default=False)

    parser.add_argument('-p', '--pool', type=utils.str2bool,
                        help=r'Run patch extraction on a pool of workers.',
                        default=True)

    parser.add_argument('-n', '--num-workers', type=int,
                        help=r'Number of pool workers (Default is OS (n-1) specified)',
                        default=os.cpu_count() - 1)

    parser.add_argument('-d', '--dry', type=utils.str2bool,
                        help=r'Do a dry run.',
                        default=False)

    arguments = parser.parse_args()

    if arguments.dry:
        print('Doing Dry Run.\n')

    if arguments.svs_listing:
        svs_listing_file = open(arguments.svs_listing).read().splitlines()
    else:
        svs_listing_file = None

    if arguments.pool:
        print('Running patch extraction on Pool of {} Workers\n'.format(arguments.num_workers))
        main_pooled(specified_svs_files=svs_listing_file, args=arguments)
    else:
        print('Running patch extraction sequentially.\n')
        main(specified_svs_files=svs_listing_file, args=arguments)

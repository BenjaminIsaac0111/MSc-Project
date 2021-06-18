import os
from multiprocessing import Pool
import argparse
from functools import partial
from utils import utils
from Processing import PatchExtractor
from pathlib import Path


# TODO Double check how to choose appropriate image level for downsampling.
def main(specified_svs_files=None, args=None):
    if args.config:
        patch_extractor = PatchExtractor(config_file=args.config)
    else:
        patch_extractor = PatchExtractor()

    if specified_svs_files:
        patch_extractor.svs_files = specified_svs_files

    # TODO Extend to extract points from multiple associated files per svs?
    for file in patch_extractor.svs_files:
        patch_extractor.load_svs(file)
        patch_extractor.load_associated_file()
        patch_extractor.extract_points()
        patch_extractor.extract_patches(dry=args.dry)
        patch_extractor.close_svs()


def main_pooled(args=None):
    extractor = PatchExtractor(config_file=args.config)
    svs_files = utils.split_list(lst=extractor.svs_files, n=round(len(extractor.svs_files) / args.num_workers))
    pool = Pool(args.num_workers)
    main_pool = partial(main, args=args)
    pool.map_async(main_pool, list(svs_files))
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the whole slide images (WSL) patch extractor")

    parser.add_argument("-c", "--config", type=Path,
                        help=r'The extractor config file (YAML). Will use the default_configuration.yaml if none '
                             r'supplied.',
                        default='config\\default_configuration.yaml')

    parser.add_argument('-a', '--assoc-file-pattern', type=str,
                        help='Regular Expression to load the annotation file(s)',
                        default=None)

    parser.add_argument('-p', '--pool', type=utils.str2bool,
                        help='Run patch extraction on a pool of workers.',
                        default=False)

    parser.add_argument('-n', '--num-workers', type=int,
                        help='Number of pool workers (Default is OS (n-1) specified)',
                        default=os.cpu_count()-1)

    parser.add_argument('-d', '--dry', type=utils.str2bool,
                        help='Do a dry run.',
                        default=False)

    arguments = parser.parse_args()

    if arguments.dry:
        print('Doing Dry Run.\n')

    if arguments.pool:
        print('Running patch extraction on Pool of {} Workers\n'.format(arguments.num_workers))
        main_pooled(args=arguments)
    else:
        print('Running patch extraction sequentially\n.')
        main(args=arguments)

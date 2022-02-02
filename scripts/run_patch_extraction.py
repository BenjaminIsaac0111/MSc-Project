import argparse
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from SVSLoader.Processing import PatchExtractor
from SVSLoader.Utils import utils


def main(specified_svs_files=None, args=None):
    if args.config:
        patch_extractor = PatchExtractor(config_file=args.config)
    else:
        patch_extractor = PatchExtractor()

    if specified_svs_files:
        patch_extractor.svs_files = specified_svs_files

    for file in patch_extractor.svs_files:
        patch_extractor.load_svs_by_id(file)
        patch_extractor.load_associated_file()
        try:
            patch_extractor.extract_points()
            patch_extractor.extract_patches()
        except AttributeError as e:
            print('\tNo associated file loaded for {}. '.format(patch_extractor.svs_id))
            print('\tCheck RegEx pattern or Missing File?\n'.format())
            continue
        finally:
            patch_extractor.close_svs()
    print('Done!')


def main_pooled(specified_svs_files=None, args=None):
    extractor = PatchExtractor(config_file=args.config)
    if specified_svs_files:
        extractor.svs_files = specified_svs_files
    svs_files = utils.list_to_blocks(lst=extractor.svs_files,
                                     n_blocks=round(len(extractor.svs_files) / args.num_workers))
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
                        default=int(os.cpu_count() / 2) - 1)

    arguments = parser.parse_args()

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

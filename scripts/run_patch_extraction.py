import os
import argparse
import inspect
from functools import partial
from multiprocessing import Pool
import numpy as np
from pathlib import Path
from SVSLoader.Loaders.svsloader import SVSLoader
from SVSLoader.Utils.utils import str2bool
from SVSLoader.Config import load_config
from importlib import import_module


def main(specified_svs_files=None, args=None):
    config = load_config(args.config)
    if config['EXTRACTION_MODULE']:
        print(f'{os.getpid()}: Using {config["EXTRACTION_MODULE"]} module to extract patches.')
        module = config["EXTRACTION_MODULE"]
    else:
        module = arguments.extractor_module

    globals()[module] = import_module(f'SVSLoader.Processing.{module.lower()}')
    members = inspect.getmembers(globals()[module], inspect.isclass)
    member = [m[0] for m in members if m[0].lower() == module.lower()][0]
    extractor = getattr(globals()[module], member)(config)

    if specified_svs_files:
        extractor.svs_files = specified_svs_files
    extractor.run_extraction()

    print(f'--- Process ID: {os.getpid()} --- Complete!')


def main_pooled(specified_svs_files=None, args=None):
    svs_search = SVSLoader(config=args.config)
    if specified_svs_files:
        svs_search.svs_files = specified_svs_files
    svs_files = np.array_split(svs_search.svs_files, args.num_workers)
    svs_files = [list(chunk) for chunk in svs_files]
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
                        default=Path('config\\default_configuration.yaml'))

    parser.add_argument('-a', '--assoc-file-pattern', type=str,
                        help=r'Regular Expression to load the annotation file(s)',
                        default=None)

    parser.add_argument('-s', '--wsi-listing', type=str,
                        help=r'Extract from images in this WSI file listing.',
                        default=False)

    parser.add_argument('-p', '--pool', type=str2bool,
                        help=r'Run patch extraction on a pool of workers.',
                        default=True)

    parser.add_argument('-n', '--num-workers', type=int,
                        help=r'Number of pool workers (Default is OS (n-1) specified)',
                        default=int(os.cpu_count() / 2) - 1)

    parser.add_argument('-e', '--extractor-module', type=str,
                        help=r'The name of the patch extractor to be run.',
                        default=None)

    arguments = parser.parse_args()

    if arguments.wsi_listing:
        wsi_listing_file = open(arguments.wsi_listing).read().splitlines()
    else:
        wsi_listing_file = None

    if arguments.pool:
        print('Running patch extraction on Pool of {} Workers\n'.format(arguments.num_workers))
        main_pooled(specified_svs_files=wsi_listing_file, args=arguments)
    else:
        print('Running patch extraction sequentially.\n')
        main(specified_svs_files=wsi_listing_file, args=arguments)

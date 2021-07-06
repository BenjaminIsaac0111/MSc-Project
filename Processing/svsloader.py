import os
from os.path import exists
from config import load_config
from utils import utils
import re

import openslide


class SVSLoader:
    def __init__(self, config_file='config\\default_configuration.yaml'):
        self.CONFIG = load_config(file=config_file)
        self.CACHE_DIRECTORY = self.CONFIG['CACHE']
        self.DATA_DIR = self.CONFIG['DATA_DIR']
        self.DIR_LISTING_FILENAME = 'directory_listing.txt'
        self.CACHED_LISTINGS = self.CACHE_DIRECTORY + self.DIR_LISTING_FILENAME
        self.svs_files = []
        self.directory_listing = []
        self.associated_files = []
        self._loaded_svs = None
        self._loaded_associated_file = None
        self._svs_name = ''
        self.construct_dir_listing()
        self.construct_svs_list()

    def construct_dir_listing(self):
        if not exists(self.CACHED_LISTINGS):
            for root, dirs, files in os.walk(self.DATA_DIR):
                for file in files:
                    self.directory_listing.append(os.path.join(root, file))
            self.cache_dir_listing()
        else:
            self.directory_listing = utils.open_dir_listings(directory_listings=self.CACHED_LISTINGS)

    def cache_dir_listing(self):
        utils.write_dir_listing(dir_listings=self.directory_listing,
                                file=self.CACHE_DIRECTORY + self.DIR_LISTING_FILENAME)

    def construct_svs_list(self):
        self.svs_files = [os.path.split(file)[-1:][0] for file in self.directory_listing if file.endswith('.svs')]

    def load_svs(self, filename=None, silent=False):
        self._loaded_svs = openslide.OpenSlide(filename=self.get_svs(pattern=filename))
        if self._loaded_svs is None:
            raise FileNotFoundError
        self._svs_name = filename
        self.associated_files = self.get_associated_files(pattern=filename[:-4])
        if not silent:
            self.loader_message()

    def load_associated_file(self, pattern=None):
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        compiled = re.compile(pattern=pattern)
        file = [file for file in self.associated_files if compiled.search(file.lower())]
        if len(file) == 1:
            path = file[0]
            file = open(file=path)
            print('\tUsing Loaded {}\n'.format(os.path.split(path)[-1]))
            self._loaded_associated_file = file
        else:
            print('\tNo associated file loaded for {}. '.format(self._svs_name) + 'Check pattern: "{}"'.format(
                pattern))

    def close_svs(self):
        self._loaded_svs.close()

    def get_svs(self, pattern):
        return [path for path in self.directory_listing if re.search(pattern, path) and path.endswith('.svs')][0]

    def get_associated_files(self, pattern=None):
        files = [path for path in self.directory_listing if
                 re.search(pattern.lower(), path.lower()) and not path.endswith('.svs')]
        return files

    def search_directory_listing(self, pattern=None):
        return [path for path in self.directory_listing if re.search(pattern, path)]

    def loader_message(self):
        print('--- Loaded {} with {} associated file(s) on PID {} ---'.format(self._svs_name,
                                                                              len(self.associated_files),
                                                                              os.getpid()))

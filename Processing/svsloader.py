import os
from config import load_config
import re

os.add_dll_directory('C:\\Program Files\\Openslide\\bin')  # Fix for Openslide bin not being found on path
from openslide import OpenSlide


class SVSLoader:
    def __init__(self, config_file='config\\default_configuration.yaml'):
        self.CONFIG = load_config(file=config_file)
        self.DATA_DIR = self.CONFIG['DATA_DIR']
        self.svs_files = []
        self.directory_listing = []
        self.associated_files = []
        self.loaded_svs = None
        self.loaded_associated_file = None
        self.svs_name = ''
        self.construct_dir_listing()
        self.construct_svs_list()

    def construct_dir_listing(self):
        for root, dirs, files in os.walk(self.DATA_DIR):
            for file in files:
                self.directory_listing.append(os.path.join(root, file))

    def construct_svs_list(self):
        self.svs_files = [os.path.split(file)[-1:][0] for file in self.directory_listing if file.endswith('.svs')]

    def load_svs(self, filename=None, silent=False):
        self.loaded_svs = OpenSlide(filename=self.get_svs(pattern=filename))
        if self.loaded_svs is None:
            raise FileNotFoundError
        self.svs_name = filename
        self.associated_files = self.get_associated_files(pattern=filename[:-4])
        if not silent:
            self.loader_message()

    def load_associated_file(self, pattern=None):
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        compiled = re.compile(pattern=pattern)

        for i, file_path in enumerate(self.associated_files):
            if compiled.search(file_path.split("\\")[-1].lower()):
                associated_file = self.associated_files[i]
                self.loaded_associated_file = open(file=associated_file)
                print('\tUsing Loaded {}\n'.format(os.path.split(associated_file)[-1]))

    def close_svs(self):
        self.loaded_svs.close()

    def get_svs(self, pattern):
        return [path for path in self.directory_listing if re.search(pattern, path) and path.endswith('.svs')][0]

    def get_associated_files(self, pattern=None):
        files = [path for path in self.directory_listing if
                 re.search(pattern.lower(), path.lower()) and not path.endswith('.svs')]
        return files

    def search_directory_listing(self, pattern=None):
        return [path for path in self.directory_listing if re.search(pattern, path)]

    def loader_message(self):
        print('--- Loaded {} with {} associated file(s) on PID {} ---'.format(self.svs_name,
                                                                              len(self.associated_files),
                                                                              os.getpid()))

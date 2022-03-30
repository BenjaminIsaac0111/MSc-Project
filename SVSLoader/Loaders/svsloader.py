import os
import re
from SVSLoader.Config import load_config
from tiffslide import TiffSlide
import pathlib


class SVSLoader:
    def __init__(self, config=None):
        if issubclass(type(config), pathlib.PurePath):
            self.CONFIG = load_config(config)
        elif type(config) == str:
            self.CONFIG = load_config(config)
        elif type(config) == dict:
            self.CONFIG = config
        self.DATA_DIR = self.CONFIG['WSL_DATA_DIR']
        self.svs_files = []
        self.directory_listing = []
        self.loaded_svs = None
        self.loaded_associated_file = None
        self.svs_id = ''
        self.institute_id = None
        self.construct_dir_listing()
        self.construct_svs_files_list()
        self.no_assoc_files_counted = 0
        self.loader_message = f''

    def construct_dir_listing(self):
        for root, dirs, files in os.walk(self.DATA_DIR):
            for file in files:
                self.directory_listing.append(os.path.join(root, file))

    def construct_svs_files_list(self):
        self.svs_files = [os.path.split(file)[-1:][0] for file in self.directory_listing if file.endswith('.svs')]

    def load_svs_by_id(self, svs_id=None):
        svs_path = self.find_svs_path_by_id(pattern=svs_id)
        self.loaded_svs = TiffSlide(filename=svs_path)
        if self.loaded_svs is None:
            raise FileNotFoundError
        self.svs_id = svs_id
        self.extract_institute_id()
        self.loader_message += f'--- Loaded {self.svs_id} on PID {os.getpid()} ---\n'

    def load_associated_file(self, pattern=None):
        self.loaded_associated_file = None
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        for file_path in self.search_directory_listing(pattern=pattern):
            if re.search(self.svs_id[:-4], os.path.split(file_path)[-1].lower()):
                self.loaded_associated_file = open(file=file_path)
                self.loader_message += f'\tUsing Loaded {self.loaded_associated_file.name}\n'
                break

    def close_svs(self):
        self.loaded_svs.close()

    def find_svs_path_by_id(self, pattern):
        return [path for path in self.directory_listing if re.search(pattern, path) and path.endswith('.svs')][0]

    def search_directory_listing(self, pattern=None):
        compiled = re.compile(pattern=pattern)
        found_files = []
        for i, file_path in enumerate(self.directory_listing):
            if compiled.search(os.path.split(file_path)[-1].lower()):
                found_files.append(self.directory_listing[i])
        return found_files

    def print_loader_message(self):
        print(self.loader_message)

    def build_patch_filenames(self):
        raise NotImplementedError

    def parse_annotation(self):
        raise NotImplementedError

    def read_patch_region(self):
        raise NotImplementedError

    def build_ground_truth_mask(self):
        raise NotImplementedError

    def build_patch(self):
        raise NotImplementedError

    def save_patch(self):
        raise NotImplementedError

    def extract_institute_id(self):
        raise NotImplementedError

    def run_patch_extraction(self):
        raise NotImplementedError

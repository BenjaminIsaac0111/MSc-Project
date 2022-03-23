import os
import re
from SVSLoader.Config import load_config

# TODO Migrate to Tiffslide.
try:
    os.add_dll_directory('C:\\Program Files\\Openslide\\bin')  # Fix for Openslide bin not being found on path
    from openslide import OpenSlide
except AttributeError:
    print('Warning: Openslide DLL fix did not complete.')


class SVSLoader:
    def __init__(self, config=None):
        if type(config) == str:
            self.CONFIG = load_config(config)
        elif type(config) == dict:
            self.CONFIG = config
        self.DATA_DIR = self.CONFIG['DATA_DIR']
        self.svs_files = []
        self.directory_listing = []
        self.loaded_svs = None
        self.loaded_associated_file = None
        self.svs_id = ''
        self.institute_id = None
        self.construct_dir_listing()
        self.construct_svs_files_list()
        self.no_assoc_files_counted = 0

    def construct_dir_listing(self):
        for root, dirs, files in os.walk(self.DATA_DIR):
            for file in files:
                self.directory_listing.append(os.path.join(root, file))

    def construct_svs_files_list(self):
        self.svs_files = [os.path.split(file)[-1:][0] for file in self.directory_listing if file.endswith('.svs')]

    def load_svs_by_id(self, svs_id=None):
        svs_path = self.find_svs_path_by_id(pattern=svs_id)
        self.loaded_svs = OpenSlide(filename=svs_path)
        if self.loaded_svs is None:
            raise FileNotFoundError
        self.svs_id = svs_id
        self.extract_institute_id()
        self._loader_message()

    def load_associated_file(self, pattern=None):
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        for file_path in self.search_directory_listing(pattern=pattern):
            if re.search(self.svs_id[:-4], os.path.split(file_path)[-1].lower()):
                self.loaded_associated_file = open(file=file_path)
                print('\tUsing Loaded {}\n'.format(self.loaded_associated_file.name))
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

    def build_patch_filenames(self):
        raise NotImplementedError

    def parse_annotation(self):
        raise NotImplementedError

    def read_patch_region(self):
        raise NotImplementedError

    def build_mask(self):
        raise NotImplementedError

    def extract_patch(self):
        raise NotImplementedError

    def save_patch(self):
        raise NotImplementedError

    def _loader_message(self):
        raise NotImplementedError

    def extract_institute_id(self):
        raise NotImplementedError


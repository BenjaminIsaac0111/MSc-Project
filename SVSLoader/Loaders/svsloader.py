import os
import re
from SVSLoader.Config import load_config

# TODO this needs tidying. Will find required dlls and add them to the project.
try:
    os.add_dll_directory('C:\\Program Files\\Openslide\\bin') # Fix for Openslide bin not being found on path
    from openslide import OpenSlide
except AttributeError:
    print('Warning: Openslide DDL fix did not complete.')


class SVSLoader:
    def __init__(self, config_file='config\\default_configuration.yaml'):
        self.CONFIG = load_config(file=config_file)
        self.DATA_DIR = self.CONFIG['DATA_DIR']
        self.svs_files = []
        self.directory_listing = []
        self.associated_files = []
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
        self.set_svs_institute()
        self._loader_message()

    def load_associated_file(self, pattern=None):
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        for file_path in self.search_directory_listing(pattern=pattern):
            # If the svs id matches the id pattern in
            if re.search(self.svs_id[:-4], os.path.split(file_path)[-1].lower()):
                self.loaded_associated_file = open(file=file_path)
                print('\tUsing Loaded {}\n'.format(self.loaded_associated_file.name))
                break

    def find_svs_path_by_id(self, pattern):
        return [path for path in self.directory_listing if re.search(pattern, path) and path.endswith('.svs')][0]

    def _loader_message(self):
        pass

    def set_svs_institute(self):
        pass

    def close_svs(self):
        self.loaded_svs.close()

    def get_associated_files(self, pattern=None):
        files = [path for path in self.directory_listing if
                 re.search(pattern.lower(), path.lower()) and not path.endswith('.svs')]
        return files

    def search_directory_listing(self, pattern=None):
        compiled = re.compile(pattern=pattern)
        found_files = []
        for i, file_path in enumerate(self.directory_listing):
            if compiled.search(os.path.split(file_path)[-1].lower()):
                found_files.append(self.directory_listing[i])
        return found_files

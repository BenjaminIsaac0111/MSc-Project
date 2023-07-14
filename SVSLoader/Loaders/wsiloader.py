import os
import re
from functools import lru_cache
import numpy as np
from SVSLoader.Config import load_config
from tiffslide import TiffSlide
import pathlib


class WSILoader:
    """
    WSILoader is a class that provides functionality to load Whole Slide Image (WSI) files and associated files.

    Attributes:
        CONFIG (dict): Configuration dictionary loaded from a configuration file.
        DATA_DIR (str): Directory path where the WSI files and associated files are located.
        image_format (str): Format of the whole slide images, defaults to '.svs'.
        whole_slide_image_filenames (list): List of filenames of the whole slide images.
        directory_listing (list): List of all file paths in the DATA_DIR directory.
        loaded_svs (TiffSlide): TiffSlide object representing the loaded WSI file.
        loaded_associated_files (list): List of opened associated files.
        whole_silde_image_id (str): ID of the loaded whole slide image.
        batch_id (None or int): ID of the batch (not used in the provided code).
        no_assoc_files_counted (int): Count of associated files (not used in the provided code).
        loader_message (str): Message containing loader information.

    Methods:
        __init__(self, configuration=None, data_dir=None):
            Initializes the WSILoader object.
            Args:
                configuration (str or dict or pathlib.PurePath): Configuration for WSILoader.
                    It can be a path to a configuration file, a dictionary containing configuration,
                    or a pathlib.PurePath object representing a configuration file path.
                data_dir (str): Directory path where the WSI files and associated files are located.

        construct_dir_listing(self):
            Constructs the directory listing by recursively walking through the DATA_DIR directory
            and adding all file paths to the directory_listing attribute.

        construct_whole_slide_image_list(self):
            Constructs the list of whole slide image filenames by filtering the directory_listing
            for files with the .svs extension and extracting the filenames.

        load_svs_by_id(self, svs_id=None):
            Loads the WSI file specified by svs_id.
            Args:
                svs_id (str): ID of the WSI file to be loaded.

        load_associated_files(self, pattern=None):
            Loads associated files based on the given pattern.
            Args:
                pattern (str): Pattern to match associated files.

        get_whole_slide_image_resolution(self, as_np_array=False):
            Returns the dimensions of the loaded whole slide image.
            Args:
                as_np_array (bool): If True, returns the dimensions as a NumPy array.
                    Otherwise, returns a tuple of width and height.

        close_svs(self):
            Closes the loaded WSI file.

        find_svs_path_by_id(self, pattern):
            Finds the file path of the WSI file based on the given pattern.
            Args:
                pattern (str): Pattern to match the WSI file.

        search_directory_listing(self, pattern=None):
            Searches the directory listing for file paths that match the given pattern.
            Args:
                pattern (str): Pattern to match the file paths.
            Returns:
                List of file paths that match the pattern.

        print_loader_message(self):
            Prints the loader message and resets the loader_message attribute.
    """

    def __init__(self, configuration=None, data_dir=None):
        """
        Initializes the WSILoader object.

        Args:
            configuration (str or dict or pathlib.PurePath): Configuration for WSILoader.
                It can be a path to a configuration file, a dictionary containing configuration,
                or a pathlib.PurePath object representing a configuration file path.
            data_dir (str): Directory path where the WSI files and associated files are located.
        """
        if issubclass(type(configuration), pathlib.PurePath):
            self.CONFIG = load_config(configuration)
        elif type(configuration) == str:
            self.CONFIG = load_config(configuration)
        elif type(configuration) == dict:
            self.CONFIG = configuration
        self.DATA_DIR = self.CONFIG['WSL_DATA_DIR']
        if data_dir:
            self.DATA_DIR = data_dir

        self.image_format = '.svs'
        self.whole_slide_image_filenames = []
        self.directory_listing = []
        self.loaded_svs = None
        self.loaded_associated_files = []
        self.whole_silde_image_id = ''
        self.batch_id = None
        self.construct_dir_listing()
        self.construct_whole_slide_image_list()
        self.no_assoc_files_counted = 0
        self.loader_message = f''

    def construct_dir_listing(self):
        """
        Constructs the directory listing by recursively walking through the DATA_DIR directory
        and adding all file paths to the directory_listing attribute.
        """
        for root, dirs, files in os.walk(self.DATA_DIR):
            for file in files:
                self.directory_listing.append(os.path.join(root, file))

    def construct_whole_slide_image_list(self):
        """
        Constructs the list of whole slide image filenames by filtering the directory_listing
        for files with the .svs extension and extracting the filenames.
        """
        self.whole_slide_image_filenames = [
            os.path.split(file)[-1:][0] for file in self.directory_listing if file.endswith('.svs')
        ]

    def load_svs_by_id(self, svs_id=None):
        """
        Loads the WSI file specified by svs_id.

        Args:
            svs_id (str): ID of the WSI file to be loaded.
        """
        svs_path = self.find_svs_path_by_id(pattern=svs_id)
        self.loaded_svs = TiffSlide(filename=svs_path)
        if self.loaded_svs is None:
            raise FileNotFoundError
        self.whole_silde_image_id = svs_id
        self.loader_message += f'--- Loaded {self.whole_silde_image_id} on PID {os.getpid()} ---\n'

    def load_associated_files(self, pattern=None):
        """
        Loads associated files based on the given pattern.

        Args:
            pattern (str): Pattern to match associated files.
        """
        self.loaded_associated_files = []
        if not pattern:
            pattern = self.CONFIG['ASSOCIATED_FILE_PATTERN']
        for file_path in self.search_directory_listing(pattern=pattern):
            if re.search(self.whole_silde_image_id[:-4], os.path.split(file_path)[-1].lower()):
                _loaded_file = open(file=file_path)
                assert self.whole_silde_image_id[:-4] == os.path.split(_loaded_file.name)[-1].split('_')[0]
                self.loaded_associated_files.append(_loaded_file)
                self.loader_message += f'\tUsing Loaded {_loaded_file.name}\n'
        return

    def get_whole_slide_image_resolution(self, as_np_array=False):
        """
        Returns the dimensions of the loaded whole slide image.

        Args:
            as_np_array (bool): If True, returns the dimensions as a NumPy array.
                Otherwise, returns a tuple of width and height.

        Returns:
            tuple or numpy.ndarray: Dimensions of the loaded whole slide image.
        """
        if not as_np_array:
            return self.loaded_svs.dimensions
        return np.array(self.loaded_svs.dimensions)

    def close_svs(self):
        """
        Closes the loaded WSI file.
        """
        self.loaded_svs.close()

    def find_svs_path_by_id(self, pattern):
        """
        Finds the file path of the WSI file based on the given pattern.

        Args:
            pattern (str): Pattern to match the WSI file.

        Returns:
            str: File path of the matched WSI file.
        """
        return [path for path in self.directory_listing if re.search(pattern, path) and path.endswith('.svs')][0]

    @lru_cache
    def search_directory_listing(self, pattern=None):
        """
        Searches the directory listing for file paths that match the given pattern.

        Args:
            pattern (str): Pattern to match the file paths.

        Returns:
            list: List of file paths that match the pattern.
        """
        compiled = re.compile(pattern=pattern)
        found_files = []
        for i, file_path in enumerate(self.directory_listing):
            if compiled.search(os.path.split(file_path)[-1].lower()):
                found_files.append(self.directory_listing[i])
        return found_files

    def print_loader_message(self):
        """
        Prints the loader message and resets the loader_message attribute.
        """
        print(self.loader_message)
        self.loader_message = f''

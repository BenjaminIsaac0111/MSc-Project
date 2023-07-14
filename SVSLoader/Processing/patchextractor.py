from abc import abstractmethod
from abc import ABCMeta
from SVSLoader.Loaders.wsiloader import WSILoader


class PatchExtractor(WSILoader, metaclass=ABCMeta):
    """
    PatchExtractor is an abstract base class that extends WSILoader and provides an interface for patch extraction
    from Whole Slide Images (WSI).

    Attributes:
        scaling_factor (float): Scaling factor for patch extraction.
        patches_dir_ (str): Directory path where the extracted patches will be saved.
        patch_w_h (tuple): Width and height of the patch.
        patch_w_h_scaled (list): Scaled width and height of the patch.
        patch (None or ndarray): Extracted patch image.
        point_index (None or int): Index of the selected point.
        points_coordinates (list): List of coordinates of the selected points.
        patch_coordinates (list): List of coordinates for each extracted patch.
        patch_classes (list): List of classes for each extracted patch.
        selected_patch_class (None or str): Selected patch class.
        patch_filenames (list): List of filenames for each extracted patch.
        loaded_wsi_region (None or Region): Loaded WSI region.

    Methods:
        __init__(self, configuration=None):
            Initializes the PatchExtractor object.
            Args:
                configuration (str or dict or pathlib.PurePath): Configuration for PatchExtractor.
                    It can be a path to a configuration file, a dictionary containing configuration,
                    or a pathlib.PurePath object representing a configuration file path.

        build_patch_filenames(self):
            Abstract method for building patch filenames.

        extract_institute_id(self):
            Abstract method for extracting the institute ID.

        read_patch_region(self):
            Abstract method for reading the patch region.

        build_patch(self):
            Abstract method for building the patch.

        save_patch(self):
            Abstract method for saving the patch.

        run_extraction(self):
            Abstract method for running the patch extraction.
    """

    def __init__(self, configuration=None):
        """
        Initializes the PatchExtractor object.

        Args:
            configuration (str or dict or pathlib.PurePath): Configuration for PatchExtractor.
                It can be a path to a configuration file, a dictionary containing configuration,
                or a pathlib.PurePath object representing a configuration file path.
        """
        super().__init__(configuration=configuration)
        self.scaling_factor = self.CONFIG['SCALING_FACTOR']
        self.patches_dir_ = self.CONFIG['PATCHES_DIR']
        self.patch_w_h = self.CONFIG['PATCH_SIZE']
        self.patch_w_h_scaled = [v * self.scaling_factor for v in self.patch_w_h]
        self.patch = None
        self.point_index = None
        self.points_coordinates = []
        self.patch_coordinates = []
        self.patch_classes = []
        self.selected_patch_class = None
        self.patch_filenames = []
        self.loaded_wsi_region = None

    @abstractmethod
    def build_patch_filenames(self):
        """
        Abstract method for building patch filenames.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_institute_id(self):
        """
        Abstract method for extracting the institute ID.
        """
        raise NotImplementedError

    @abstractmethod
    def read_patch_region(self):
        """
        Abstract method for reading the patch region.
        """
        raise NotImplementedError

    @abstractmethod
    def build_patch(self):
        """
        Abstract method for building the patch.
        """
        raise NotImplementedError

    @abstractmethod
    def save_patch(self):
        """
        Abstract method for saving the patch.
        """
        raise NotImplementedError

    @abstractmethod
    def run_extraction(self):
        """
        Abstract method for running the patch extraction.
        """
        raise NotImplementedError

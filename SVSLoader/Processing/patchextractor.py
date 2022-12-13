from abc import abstractmethod
from abc import ABCMeta

from SVSLoader.Loaders.svsloader import SVSLoader


class PatchExtractor(SVSLoader, metaclass=ABCMeta):
    def __init__(self, configuration=None):
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
        raise NotImplementedError

    @abstractmethod
    def extract_institute_id(self):
        raise NotImplementedError

    @abstractmethod
    def read_patch_region(self):
        raise NotImplementedError

    @abstractmethod
    def build_patch(self):
        raise NotImplementedError

    @abstractmethod
    def save_patch(self):
        raise NotImplementedError

    @abstractmethod
    def run_extraction(self):
        raise NotImplementedError

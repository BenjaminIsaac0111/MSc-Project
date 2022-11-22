from abc import abstractmethod

from SVSLoader.Loaders.svsloader import SVSLoader


class PatchExtractor(SVSLoader):
    def __init__(self, config_file=None):
        super().__init__(config_file=config_file)
        self._SCALING_FACTOR = self.CONFIG['SCALING_FACTOR']  # TODO compute using function based on mag input?
        self.patches_dir_ = self.CONFIG['PATCHES_DIR']
        self.patch_w_h = self.CONFIG['PATCH_SIZE']
        self.patch_w_h_scaled = [v * self._SCALING_FACTOR for v in self.patch_w_h]
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
    def read_patch_region(self, loc_idx):
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

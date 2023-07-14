import os
from abc import ABC
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image
from SVSLoader.Processing.patchextractor import PatchExtractor


class TileExtractor(PatchExtractor, ABC):
    """
    TileExtractor is a concrete class that extends PatchExtractor and provides functionality for extracting tiles
    from Whole Slide Images (WSI).

    Attributes:
        _SCALING_FACTOR (float): Scaling factor for tile extraction.
        patches_dir_ (str): Directory path where the extracted tiles will be saved.
        patch_w_h (tuple): Width and height of the tile patch.
        patch_w_h_scaled (list): Scaled width and height of the tile patch.
        patch_idx (None or int): Index of the current tile patch.
        patch_coordinates (list): List of coordinates for each tile patch.
        patch_filenames (list): List of filenames for each tile patch.
        loaded_rgb_patch_img (None or ndarray): Loaded RGB patch image.

    Methods:
        __init__(self, configuration=None):
            Initializes the TileExtractor object.
            Args:
                configuration (str or dict or pathlib.PurePath): Configuration for TileExtractor.
                    It can be a path to a configuration file, a dictionary containing configuration,
                    or a pathlib.PurePath object representing a configuration file path.

        read_patch_region(self, patch_idx=None):
            Reads the region of the tile patch specified by patch_idx.
            Args:
                patch_idx (int): Index of the tile patch to be read.

        build_patch_filenames(self):
            Builds the filenames for each tile patch.

        save_patch(self):
            Saves the current tile patch.

        run_extraction(self):
            Runs the tile extraction process.

        build_meshgrid_coordinates(self):
            Builds the meshgrid coordinates for tile extraction.

        extract_batch_id(self):
            Extracts the batch ID from the WSI filename.
    """

    def __init__(self, configuration=None):
        """
        Initializes the TileExtractor object.

        Args:
            configuration (str or dict or pathlib.PurePath): Configuration for TileExtractor.
                It can be a path to a configuration file, a dictionary containing configuration,
                or a pathlib.PurePath object representing a configuration file path.
        """
        super().__init__(configuration=configuration)
        self._SCALING_FACTOR = self.CONFIG['SCALING_FACTOR']
        self.patches_dir_ = self.CONFIG['PATCHES_DIR']
        self.patch_w_h = self.CONFIG['PATCH_SIZE']
        self.patch_w_h_scaled = [v * self._SCALING_FACTOR for v in self.patch_w_h]
        self.patch_idx = None
        self.patch_coordinates = []
        self.patch_filenames = []
        self.loaded_rgb_patch_img = None

    def read_patch_region(self, patch_idx=None):
        """
        Reads the region of the tile patch specified by patch_idx.

        Args:
            patch_idx (int): Index of the tile patch to be read.
        """
        self.patch_idx = patch_idx
        x, y = self.patch_coordinates[self.patch_idx]
        self.loaded_rgb_patch_img = self.loaded_svs.read_region(
            location=(x * self.patch_w_h[0], y * self.patch_w_h[1]),
            level=0,
            size=self.patch_w_h,
            padding=False
        )
        self.loaded_rgb_patch_img = np.array(self.loaded_rgb_patch_img.convert("RGB"))
        self.loaded_rgb_patch_img = cv.resize(
            src=self.loaded_rgb_patch_img,
            dsize=self.patch_res
        )

    def build_patch_filenames(self):
        """
        Builds the filenames for each tile patch.
        """
        self.patch_filenames = []
        for i, loc in enumerate(self.patch_coordinates):
            _patch_filename = ''
            if self.batch_id:
                _patch_filename += f'{self.batch_id[:2]}_{self.batch_id[-2:]}_'
            _patch_filename += f'{self.whole_silde_image_id[:-4]}_{str(i)}_'
            _patch_filename += f'{loc[0]}_{loc[1]}.png'
            self.patch_filenames.append(_patch_filename)

    def save_patch(self):
        """
        Saves the current tile patch.
        """
        Image.fromarray(self.loaded_rgb_patch_img).save(
            fp=self.patches_dir_ + self.patch_filenames[self.patch_idx]
        )

    def run_extraction(self):
        """
        Runs the tile extraction process.
        """
        _existing = [file for file in os.listdir(self.CONFIG['PATCHES_DIR']) if file.endswith('.png')]
        for i, file in enumerate(self.whole_slide_image_filenames):
            self.load_svs_by_id(file)
            self.build_meshgrid_coordinates()
            self.build_patch_filenames()
            self.loader_message += f'\tExtracting {len(self.patch_filenames)} tiles...\n'
            self.print_loader_message()
            for j, filename in enumerate(self.patch_filenames):
                if filename not in _existing:
                    self.read_patch_region(patch_idx=j)
                    self.save_patch()
                    continue

    def build_meshgrid_coordinates(self):
        """
        Builds the meshgrid coordinates for tile extraction.
        """
        x_n_y_n_tiles = np.floor(self.get_whole_slide_image_resolution() / np.array(self.patch_w_h)).astype('uint8')
        mg = np.array(
            np.meshgrid(
                np.arange(x_n_y_n_tiles[0]),  # i
                np.arange(x_n_y_n_tiles[1])  # j
            )
        )
        self.patch_coordinates = mg.reshape(2, np.prod(x_n_y_n_tiles)).T  # ij

    def extract_batch_id(self):
        """
        Extracts the batch ID from the WSI filename.
        """
        self.batch_id = Path(self.find_svs_path_by_id(pattern=self.whole_silde_image_id)).parts[-2]

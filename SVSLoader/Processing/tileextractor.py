import os
from abc import ABC
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image
from SVSLoader.Processing.patchextractor import PatchExtractor


class TileExtractor(PatchExtractor, ABC):
    def __init__(self, configuration=None):
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
        self.patch_filenames = []
        for i, loc in enumerate(self.patch_coordinates):
            _patch_filename = ''
            if self.batch_id:
                _patch_filename += f'{self.batch_id[:2]}_{self.batch_id[-2:]}_'
            _patch_filename += f'{self.svs_id[:-4]}_{str(i)}_'
            _patch_filename += f'{loc[0]}_{loc[1]}.png'
            self.patch_filenames.append(_patch_filename)

    def save_patch(self):
        Image.fromarray(self.loaded_rgb_patch_img).save(
            fp=self.patches_dir_ + self.patch_filenames[self.patch_idx]
        )

    def run_extraction(self):
        _existing = [file for file in os.listdir(self.CONFIG['PATCHES_DIR']) if file.endswith('.png')]
        for i, file in enumerate(self.svs_files):
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
        x_n_y_n_tiles = np.floor(self.get_wsi_res() / np.array(self.patch_w_h)).astype('uint8')
        mg = np.array(
            np.meshgrid(
                np.arange(x_n_y_n_tiles[0]),  # i
                np.arange(x_n_y_n_tiles[1])  # j
            )
        )
        self.patch_coordinates = mg.reshape(2, np.prod(x_n_y_n_tiles)).T  # ij

    def extract_batch_id(self):
        self.batch_id = Path(self.find_svs_path_by_id(pattern=self.svs_id)).parts[-2]

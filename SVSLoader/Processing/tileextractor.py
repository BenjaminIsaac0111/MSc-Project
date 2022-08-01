import os
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image
from SVSLoader.Loaders.svsloader import SVSLoader


class TileExtractor(SVSLoader):
    def __init__(self, config='config\\default_configuration.yaml'):
        super().__init__(config=config)
        self._SCALING_FACTOR = self.CONFIG['SCALING_FACTOR']  # TODO compute using function based on mag input?
        self.patches_dir_ = self.CONFIG['PATCHES_DIR']

        self.patch_res = (
            self.CONFIG['PATCH_SIZE']['WIDTH'],
            self.CONFIG['PATCH_SIZE']['HEIGHT']
        )
        self.patch_w_h = (
            self.CONFIG['PATCH_SIZE']['WIDTH'] * self._SCALING_FACTOR,
            self.CONFIG['PATCH_SIZE']['HEIGHT'] * self._SCALING_FACTOR
        )

        self.patch_idx = None
        self.patch_coordinates = []
        self.patch_filenames = []
        self.loaded_rgb_patch_img = None

    def extract_institute_id(self):
        self.institute_id = Path(self.find_svs_path_by_id(pattern=self.svs_id)).parts[-2]

    def read_patch_region(self, level=0, patch_idx=None):
        self.patch_idx = patch_idx
        x, y = self.patch_coordinates[self.patch_idx]
        self.loaded_rgb_patch_img = self.loaded_svs.read_region(
            location=(x * self.patch_w_h[0], y * self.patch_w_h[1]),
            level=level,
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
            if self.institute_id:
                _patch_filename += f'{self.institute_id[:2]}_{self.institute_id[-2:]}_'
            _patch_filename += f'{self.svs_id[:-4]}_{str(i)}_'
            _patch_filename += f'{loc[0]}_{loc[1]}.png'
            self.patch_filenames.append(_patch_filename)

    def save_patch(self):
        Image.fromarray(self.loaded_rgb_patch_img).save(
            fp=self.patches_dir_ + self.patch_filenames[self.patch_idx]
        )

    def run_extraction(self):
        existing = [file for file in os.listdir(self.CONFIG['PATCHES_DIR']) if file.endswith('.png')]
        for i, file in enumerate(self.svs_files):
            self.load_svs_by_id(file)
            self.build_meshgrid_coordinates()
            self.build_patch_filenames()
            self.loader_message += f'\tExtracting {len(self.patch_filenames)} tiles...\n'
            self.print_loader_message()
            for j, filename in enumerate(self.patch_filenames):
                if filename not in existing:
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

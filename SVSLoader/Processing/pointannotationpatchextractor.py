import os
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup as soup
from SVSLoader.Processing.patchextractor import PatchExtractor


class PointAnnotationPatchExtractor(PatchExtractor):
    def __init__(self, config_file=None):
        super().__init__(config_file=config_file)
        self.patch_center = self.get_patch_center()
        self.ground_truth_mask = None

    def extract_institute_id(self):
        self.batch_id = Path(self.find_svs_path_by_id(pattern=self.svs_id)).parts[-2]

    def read_patch_region(self, level=0, loc_idx=None):
        self.point_index = loc_idx
        self.loaded_wsi_region = self.loaded_svs.read_region(
            location=self.patch_coordinates[self.point_index],
            level=level,
            size=self.patch_w_h_scaled,
            padding=True
        )

        self.loaded_wsi_region = np.array(self.loaded_wsi_region.convert("RGB"))
        self.loaded_wsi_region = cv.resize(
            src=self.loaded_wsi_region,
            dsize=self.patch_w_h
        )
        self.selected_patch_class = self.patch_classes[self.point_index]

    def get_patch_center(self):
        patch_center_x, patch_center_y = self.patch_w_h_scaled
        return int(round(patch_center_x / 2)), int(round(patch_center_y / 2))

    def build_patch(self):
        self.patch = cv.hconcat([self.loaded_wsi_region, self.ground_truth_mask])

    def parse_annotation(self):
        annotation = soup(''.join(self.loaded_associated_file.readlines()), 'html.parser')
        points = annotation.findAll('region', {'type': '3'})
        points_coor = []
        patch_classes = []
        for i, point in enumerate(points):
            patch_classes.append(point['text'])
            points_coor.append((round(float(point.find('vertices').contents[1]['x'])),
                                round(float(point.find('vertices').contents[1]['y']))))
        self.points_coordinates = points_coor
        self.patch_classes = patch_classes
        self.patch_coordinates = [(int(coor[0] - (self.patch_w_h_scaled[0] / 2)),
                                   int(coor[1] - (self.patch_w_h_scaled[1] / 2))) for coor in self.points_coordinates]

    def build_patch_filenames(self):
        self.patch_filenames = []
        _filenames = []
        for i, loc in enumerate(self.patch_coordinates):
            _patch_filename = ''
            if self.batch_id:
                _patch_filename += f'{self.batch_id[-2:]}_'
            _patch_filename += f'{self.svs_id[:-4]}_{str(i)}_Class_{self.patch_classes[i]}.png'
            _filenames.append(_patch_filename)
            self.patch_filenames = _filenames

    def build_ground_truth_mask(self):
        circle_center_coor = tuple(int(coor / 2) for coor in self.patch_w_h)
        mask = np.zeros(self.loaded_wsi_region.shape, dtype=np.uint8)
        self.ground_truth_mask = cv.circle(img=mask,
                                           center=circle_center_coor,
                                           radius=self.CONFIG['CONTEXT_MASK_RADIUS'],
                                           color=(0, 0, int(self.patch_classes[self.point_index]) + 1),
                                           thickness=-1)

    def save_patch(self):
        patch_filepath = f'{self.CONFIG["PATCHES_DIR"]}\\{self.patch_filenames[self.point_index]}'
        Image.fromarray(self.patch).save(fp=patch_filepath)

    def run_extraction(self):
        if not os.path.exists(f'{self.CONFIG["PATCHES_DIR"]}'):
            os.mkdir(f'{self.CONFIG["PATCHES_DIR"]}\\')
        for i, file in enumerate(self.svs_files):
            self.load_svs_by_id(file)
            self.load_associated_file()

            try:
                self.parse_annotation()
            except AttributeError as e:
                m = f'\tNo associated file found for {file} using RegEx: {self.CONFIG["ASSOCIATED_FILE_PATTERN"]}.\n'
                self.loader_message += m
                self.print_loader_message()
                continue

            self.build_patch_filenames()

            for j, filename in enumerate(self.patch_filenames):
                self.read_patch_region(loc_idx=j)
                self.build_ground_truth_mask()
                self.build_patch()
                self.save_patch()
                self.print_loader_message()

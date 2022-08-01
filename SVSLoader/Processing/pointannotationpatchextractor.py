import os
from pathlib import Path
import cv2 as cv
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup as soup
from SVSLoader.Loaders.svsloader import SVSLoader


class PointAnnotationPatchExtractor(SVSLoader):
    def __init__(self, config='config\\default_configuration.yaml'):
        super().__init__(config=config)
        self._SCALING_FACTOR = self.CONFIG['SCALING_FACTOR']  # TODO compute using function based on mag input?
        self.patches_dir_ = self.CONFIG['PATCHES_DIR']
        self.patch_w_h = (self.CONFIG['PATCH_SIZE']['WIDTH'],
                          self.CONFIG['PATCH_SIZE']['HEIGHT'])
        self.patch_w_h_scaled = (self.CONFIG['PATCH_SIZE']['WIDTH'] * self._SCALING_FACTOR,
                                 self.CONFIG['PATCH_SIZE']['HEIGHT'] * self._SCALING_FACTOR)
        self.patch = None
        self.point_index = None
        self.points_coordinates = []
        self.patch_coordinates = []
        self.patch_classes = []
        self.selected_patch_class = None
        self.patch_filenames = []
        self.loaded_rgb_patch_img = None
        self.ground_truth_mask = None
        self.patch_center = self.get_patch_center()

    def extract_institute_id(self):
        self.institute_id = Path(self.find_svs_path_by_id(pattern=self.svs_id)).parts[-2]

    def read_patch_region(self, level=0, loc_idx=None):
        self.point_index = loc_idx
        self.loaded_rgb_patch_img = self.loaded_svs.read_region(
            location=self.patch_coordinates[self.point_index],
            level=level,
            size=self.patch_w_h_scaled,
            padding=True
        )

        self.loaded_rgb_patch_img = np.array(self.loaded_rgb_patch_img.convert("RGB"))
        self.loaded_rgb_patch_img = cv.resize(
            src=self.loaded_rgb_patch_img,
            dsize=self.patch_w_h
        )
        self.selected_patch_class = self.patch_classes[self.point_index]

    def get_patch_center(self):
        patch_center_x, patch_center_y = self.patch_w_h_scaled
        return int(round(patch_center_x / 2)), int(round(patch_center_y / 2))

    def build_patch(self):
        self.patch = cv.hconcat([self.loaded_rgb_patch_img, self.ground_truth_mask])

    def parse_annotation(self):
        annotation = soup(''.join(self.loaded_associated_file.readlines()), 'html.parser')
        points = annotation.findAll('region', {'type': '3'})
        points_coor = []
        patch_classes = []
        for i, point in enumerate(points):
            patch_classes.append(point['text'])
            points_coor.append((round(float(point.find('vertices').contents[0]['x'])),
                                round(float(point.find('vertices').contents[0]['y']))))
        self.points_coordinates = points_coor
        self.patch_classes = patch_classes
        self.patch_coordinates = [(int(coor[0] - (self.patch_w_h_scaled[0] / 2)),
                                   int(coor[1] - (self.patch_w_h_scaled[1] / 2))) for coor in self.points_coordinates]

    def build_patch_filenames(self):
        self.patch_filenames = []
        _filenames = []
        for i, loc in enumerate(self.patch_coordinates):
            _patch_filename = ''
            if self.institute_id:
                _patch_filename += f'{self.institute_id[:2]}_{self.institute_id[-2:]}_'
            _patch_filename += f'{self.svs_id[:-4]}_{str(i)}_Class_{self.patch_classes[i]}.png'
            _filenames.append(_patch_filename)
            self.patch_filenames = _filenames

    def build_ground_truth_mask(self, truth=0):
        circle_center_coor = (
            int(round(self.CONFIG['PATCH_SIZE']['WIDTH'] / 2)),
            int(round(self.CONFIG['PATCH_SIZE']['HEIGHT'] / 2))
        )
        mask = np.zeros(self.loaded_rgb_patch_img.shape, dtype=np.uint8)
        self.ground_truth_mask = cv.circle(img=mask,
                                           center=circle_center_coor,
                                           radius=self.CONFIG['CONTEXT_MASK_RADIUS'],
                                           color=(0, 0, int(self.patch_classes[self.point_index]) + 1),
                                           thickness=-1)

    def save_patch(self):
        Image.fromarray(self.patch).save(fp=self.patches_dir_ + self.patch_filenames[self.point_index])
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

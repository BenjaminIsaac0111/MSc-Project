import os
from Processing import SVSLoader
import cv2 as cv
from PIL import Image
from bs4 import BeautifulSoup as soup
import numpy as np


class PatchExtractor(SVSLoader):
    def __init__(self, config_file='config\\default_configuration.yaml'):
        super().__init__(config_file=config_file)
        self.points_coordinates = []
        self.patch_coordinates = []
        self.patch_classes = []
        self.expand_factor = self.CONFIG['EXPAND_OUT_FACTOR']
        self.patch_size = (self.CONFIG['PATCH_SIZE']['WIDTH'] * self.expand_factor,
                           self.CONFIG['PATCH_SIZE']['HEIGHT'] * self.expand_factor)
        self.patch_center = self.get_patch_center()

    def get_patch_center(self):
        patch_center_x, patch_center_y = self.patch_size
        return int(round(patch_center_x / 2)), int(round(patch_center_y / 2))

    def get_original_patch_area_size(self):
        return self.patch_size[0] / self.expand_factor, self.patch_size[1] / self.expand_factor

    def extract_patches(self, level=0, visualise_mask=False):
        for i, loc in enumerate(self.patch_coordinates):
            if visualise_mask:
                mask = self.build_mask(truth=int(self.patch_classes[i])) * 255
            else:
                mask = self.build_mask(truth=int(self.patch_classes[i]))

            png_filename = self.svs_name[:-4] + '_' + str(i) + '_Class_' + self.patch_classes[i] + '.png'

            with self.loaded_svs.read_region(location=loc, level=level, size=self.patch_size) as img_patch:
                if not os.path.isdir(self.CONFIG['PATCHES_DIR']):
                    os.makedirs(self.CONFIG['PATCHES_DIR'])
                img_patch = np.array(img_patch.convert("RGB"))
                img_patch = cv.resize(src=img_patch, dsize=(self.CONFIG['PATCH_SIZE']['WIDTH'],
                                                            self.CONFIG['PATCH_SIZE']['HEIGHT']))
                patch = Image.fromarray(cv.hconcat([img_patch, mask]))

                patch.save(fp=self.CONFIG['PATCHES_DIR'] + png_filename)

    def extract_points(self):
        annotations = soup(''.join(self.loaded_associated_file.readlines()), 'html.parser')
        points = annotations.findAll('region', {'type': '3'})

        points_coor = []
        patch_classes = []
        for i, point in enumerate(points):
            points_coor.append((round(float(point.find('vertices').contents[0]['x'])),
                                round(float(point.find('vertices').contents[0]['y'])))
                               )
            patch_classes.append(point['text'])  # NOTE: Might be some other tag for newer files.

        self.points_coordinates = points_coor
        self.patch_classes = patch_classes
        self.patch_coordinates = [(int(coor[0] - (self.patch_size[0] / 2)),
                                   int(coor[1] - (self.patch_size[1] / 2))) for coor in self.points_coordinates]

    def build_mask(self, truth=0):
        # IMPORTANT NOTE: Open CV compatibility requires inverted geometric sizes, coors and parameters.
        shape = [self.CONFIG['PATCH_SIZE']['HEIGHT'], self.CONFIG['PATCH_SIZE']['WIDTH'], 3]
        circle_center_coor = (int(round(self.CONFIG['PATCH_SIZE']['WIDTH'] / 2)),
                              int(round(self.CONFIG['PATCH_SIZE']['HEIGHT'] / 2)))
        mask = np.zeros(shape, dtype=np.uint8)
        return cv.circle(img=mask,
                         center=circle_center_coor,
                         radius=self.CONFIG['MASK']['RADIUS'],
                         color=(0, 0, truth + 1),  # Blue channel used for ground truth. +1 to avoid void masks.
                         thickness=-1)  # RGB

    def loader_message(self):
        message = '--- Loaded {} on PID {} ---'
        print(message.format(self.svs_name, os.getpid()))

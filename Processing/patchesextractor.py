import os
import matplotlib.pyplot as plt

from Processing import SVSLoader

import cv2 as cv
from PIL import Image
from bs4 import BeautifulSoup as bs
import numpy as np


class PatchExtractor(SVSLoader):
    def __init__(self, config_file='config\\default_configuration.yaml'):
        super().__init__(config_file=config_file)
        self.locations = None
        self.points_coordinates = []
        self.patch_coordinates = []
        self.patch_classes = []
        self.patch_size = [self.CONFIG['PATCH_SIZE']['WIDTH'], self.CONFIG['PATCH_SIZE']['HEIGHT']]
        self.patch_center = [int(round(self.patch_size[0] / 2)) - 1, int(round(self.patch_size[1] / 2)) - 1]

    def extract_patches(self, level=0, dry=False):
        if level:
            level = self.loaded_svs.get_best_level_for_downsample(downsample=self.CONFIG['DOWNSAMPLE_FACTOR'])
        for i, loc in enumerate(self.patch_coordinates):
            mask = self.build_mask(truth=int(self.patch_classes[i]))
            # mask = self.build_mask(truth=255)
            png_filename = self.svs_name[:-4] + '_' + str(i) + '_Class_' + self.patch_classes[i] + '.png'
            with self.loaded_svs.read_region(location=loc, level=level, size=self.patch_size) as img_patch:
                if not dry:
                    if not os.path.isdir(self.CONFIG['PATCHES_DIR']):
                        os.makedirs(self.CONFIG['PATCHES_DIR'])
                    img_patch = np.array(img_patch.convert("RGB"))
                    patch = Image.fromarray(cv.hconcat([img_patch, mask]))
                    patch.save(fp=self.CONFIG['PATCHES_DIR'] + png_filename)

    def extract_points(self):
        try:
            annotations = bs(''.join(self.loaded_associated_file.readlines()), 'html.parser')
            points = annotations.findAll('region', {'type': '3'})

            points_coor = []
            patch_classes = []
            for i, point in enumerate(points):
                points_coor.append(
                    (round(float(point.find('vertices').contents[0]['x'])),
                     round(float(point.find('vertices').contents[0]['y'])))
                )
                patch_classes.append(point['text'])  # TODO Might be some other tag for newer files.
            self.points_coordinates = points_coor
            self.patch_classes = patch_classes
            self.patch_coordinates = [(int(coor[0] - (self.patch_size[0] / 2)),
                                       int(coor[1] - (self.patch_size[1] / 2))) for coor in self.points_coordinates]

        except AttributeError:
            print('\tCannot extract points!')

    # TODO Maybe masks should be its own class or module collection? Might want mutations of the masks in different
    #  shapes in the future.
    def build_mask(self, truth=0):
        # IMPORTANT NOTE: Open CV compatibility requires inverted geometric sizes, coors and parameters.
        shape = [self.CONFIG['PATCH_SIZE']['HEIGHT'], self.CONFIG['PATCH_SIZE']['WIDTH'], 3]
        mask = np.zeros(shape, dtype=np.uint8)
        return cv.circle(img=mask,
                         center=self.patch_center,
                         radius=self.CONFIG['MASK']['RADIUS'],
                         color=(0, 0, truth+1),  # Blue channel used for ground truth. +1 to avoid void masks.
                         thickness=-1)  # RGB

    def cache_points(self):
        raise NotImplementedError

    def plot_patch(self, i, level=0):
        plt.imshow(self.loaded_svs.read_region(location=self.patch_coordinates[i],
                                               level=level,
                                               size=self.patch_size))
        plt.title('ID: ' + str(i) + ' Class: ' + self.patch_classes[i])
        plt.show()

    def loader_message(self):
        message = '--- Loaded {} on PID {} ---'
        print(message.format(self.svs_name,
                             os.getpid()))

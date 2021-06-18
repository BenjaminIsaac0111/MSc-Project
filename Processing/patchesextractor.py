import os
import matplotlib.pyplot as plt

from Processing import SVSLoader

# import opencv
from bs4 import BeautifulSoup as bs


# r'w.*scores.xml' sample regex for WHOLE_scores.xml associated files.


class PatchExtractor(SVSLoader):
    def __init__(self, config_file='config\\default_configuration.yaml'):
        super().__init__(config_file=config_file)
        self.locations = None
        self.patch_coordinates = []
        self.patch_classes = []
        self.patch_size = (self.CONFIG['PATCH_SIZE']['WIDTH'], self.CONFIG['PATCH_SIZE']['HEIGHT'])
        self.ground_truth_mask = None

    def extract_patches(self, level=None, dry=False):
        if level:
            level = self._loaded_svs.get_best_level_for_downsample(downsample=self.CONFIG['DOWNSAMPLE_FACTOR'])
        for i, loc in enumerate(self.patch_coordinates):
            with self._loaded_svs.read_region(location=loc, level=level, size=self.patch_size) as img_patch:
                png_file_path = self._svs_name[:-4] + '_' + str(i) + '_Class_' + self.patch_classes[i] + '.png'
                if not dry:
                    img_patch.save(fp=self.CONFIG['PATCHES_DIR'] + png_file_path)

    def extract_points(self, assoc_file_pattern=None):
        if assoc_file_pattern:
            self.load_associated_file(pattern=assoc_file_pattern)
        else:
            self.load_associated_file(pattern=self.CONFIG['ASSOCIATED_FILE_PATTERN'])

        try:
            annotations = bs(''.join(self._loaded_associated_file.readlines()), 'html.parser')
            points = annotations.findAll('region', {'type': '3'})

            patch_locs = []
            patch_classes = []
            for i, point in enumerate(points):
                patch_locs.append(
                    (round(float(point.find('vertices').contents[0]['x'])),
                     round(float(point.find('vertices').contents[0]['y'])))
                )
                patch_classes.append(point['text'])
            self.patch_coordinates = patch_locs
            self.patch_classes = patch_classes

        except AttributeError:
            pass

    def cache_points(self):
        raise NotImplementedError

    # TODO Centre the patches and add ground truth masking method.
    def set_ground_truth_mask(self):
        pass

    def plot_patch(self, i, level=0, ground_truth=None):
        plt.imshow(self._loaded_svs.read_region(location=self.patch_coordinates[i],
                                                level=level,
                                                size=self.patch_size))
        plt.title('ID: ' + str(i) + ' Class: ' + str(ground_truth))
        plt.show()

    def loader_message(self):
        message = '--- Extracting patches from {} on PID {} ---'
        print(message.format(self._svs_name,
                             os.getpid()))

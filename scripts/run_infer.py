import os
import warnings

from src.InferenceUtils.test_data_preparation import process_path_value_per_class, prepare_dataset
from src.InferenceUtils.tf_customs import Linear

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore')

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.losses
import logging
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, classification_report
from src.utils import utils, plotters
from config import config

tf.autograph.set_verbosity(1)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Fool pyinstaller into including code (which is used as a lambda)
GN = tfa.layers.GroupNormalization

# Set some parameters
CONFIG = config.load_config(r'config/r32px.yaml')

IMG_WIDTH = 512
IMG_HEIGHT = 1024
IMG_CHANNELS = 3
BATCH_SIZE = 3
OUT_CHANNELS = 9
COMPONENTS = ['0: Non-Inf',
              '1: Tumour',
              '2: Str/Fib',
              '3: Necr',
              '4: Vessel',
              '5: Infl',
              '6: TumLu',
              '7: Mucin',
              '8: Muscle']

TEST_BY_INSTITUTE_ID = True
MODEL_PATH = r'E:\Results\J_MODEL_24px.output.h5'
DEVICE = '/GPU:0'
PATCHES_DIR = CONFIG['PATCHES_DIR']  # Does not matter what model data, takes ground truth from center pixel of mask.

patch_test_list = open(CONFIG['TEST_DATA'])

patch_test_list = [image.partition('\t')[0] for image in patch_test_list.readlines()]

set_of_institute_ids = set([institute_id[:2] for institute_id in patch_test_list])


def load_h5_model(model_path=None):
    return load_model(model_path,
                      compile=False,
                      custom_objects={'LeakyReLU': tensorflow.keras.layers.LeakyReLU(alpha=0.01),
                                      'Linear': Linear})


def run_infer(test_list=None, model=None):
    list_ds = tf.data.Dataset.from_tensor_slices(test_list)
    list_ds_it = iter(list_ds)
    labeled_ds = list_ds.map(lambda x: process_path_value_per_class(x, OUT_CHANNELS),
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = prepare_dataset(labeled_ds, batch_size=BATCH_SIZE, cache=None, prefetch_buffer_size=16)
    batches = iter(test_ds)
    y_true = []
    y_point_pred = []
    d = 1
    for batch in batches:  # for each patch batch
        [x_test, y_test] = batch  # for each patch batch and truth
        preds_train = model.predict(x_test, verbose=0)  # Make Prediction, 9 output channels.
        preds_train_t = (preds_train * 255).astype(np.uint8)  # Scale?

        for i in range(BATCH_SIZE):
            y_true.append(np.squeeze(y_test[i, 512, 256, :]).argmax())
            y_point_pred.append(preds_train_t[i, 64, 32, :].argmax())
            print('Patch {} : Pred {} : {} of {}'.format(next(list_ds_it),
                                                         preds_train_t[i, 64, 32, :].argmax(),
                                                         d,
                                                         len(test_list)))

            d += 1
            if d > len(test_list):
                break

        print('----')
        print('Kappa Score: ', cohen_kappa_score(y_true, y_point_pred))
        print('Accuracy Score: ', accuracy_score(y_true, y_point_pred))
        print('Balanced Accuracy Score: ', balanced_accuracy_score(y_true, y_point_pred))
        print('----')

        if d > len(test_list):
            break
    return y_point_pred, y_true


with tf.device(DEVICE):
    h5 = load_h5_model(MODEL_PATH)

result_file = []
preds, truth = [], []
if TEST_BY_INSTITUTE_ID:
    for institute_id in set_of_institute_ids:
        institute_test_list = [PATCHES_DIR + patch_name for patch_name in patch_test_list if
                               patch_name[:2] == str(institute_id)]
        preds, truth = run_infer(test_list=institute_test_list, model=h5)
        result_file = MODEL_PATH + '_' + str(institute_id) + '_' + str(datetime.now()).replace(':', '.') + '.txt'
        utils.write_list([str(p) for p in preds], result_file)
        plotters.plot_confusion(y_true=truth, y_pred=preds, fmt='d', labels=COMPONENTS)
        with open(result_file[:-4] + '_Report.txt', 'w') as report:
            print('Kappa Score: ', cohen_kappa_score(truth, preds), '\n', file=report)
            print('Balanced Accuracy Score: ', balanced_accuracy_score(truth, preds), '\n', file=report)
            print(classification_report(truth, preds), file=report)
else:
    patch_test_list = [PATCHES_DIR + patch_name for patch_name in patch_test_list]
    preds, truth = run_infer(test_list=patch_test_list, model=h5)
    result_file = MODEL_PATH + '_complete_' + str(datetime.now()).replace(':', '.') + '.txt'
    utils.write_list([str(p) for p in preds], result_file)
    plotters.plot_confusion(y_true=truth, y_pred=preds, fmt='d', labels=COMPONENTS)
    with open(result_file[:-4] + '_Report.txt', 'w') as report:
        print('Kappa Score: ', cohen_kappa_score(truth, preds), '\n', file=report)
        print('Balanced Accuracy Score: ', balanced_accuracy_score(truth, preds), '\n', file=report)
        print(classification_report(truth, preds), file=report)

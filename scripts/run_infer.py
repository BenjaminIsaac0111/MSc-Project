import os
import warnings

import pandas as pd

from src.InferenceUtils.tf_customs import Linear
from src.Loaders.patchloader import process_path_value_per_class, prepare_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
# warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
# warnings.filterwarnings('ignore')

import numpy as np
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import tensorflow as tf
import tensorflow.keras.losses
import logging
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score
from config import config

tf.autograph.set_verbosity(1)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Fool pyinstaller into including code (which is used as a lambda)
GN = tfa.layers.GroupNormalization

IMG_WIDTH = 512
IMG_HEIGHT = 1024
IMG_CHANNELS = 3
BATCH_SIZE = 8
OUT_CHANNELS = 9
MODEL_PATH = r'E:\JNET_Results_1\Collated_16px\J_MODEL_16px.output.h5'
DEVICE = '/GPU:0'
PATCHES_DIR = r'E:\Complete_Working_Data\CR07\HGDL_Model_Patch_Data\JNET_EXPERIMENTS\32px\\'
patch_test_list = open(r'E:\Complete_Working_Data\CR07\HGDL_Model_Patch_Data\JNET_EXPERIMENTS\TestData.txt')
patch_test_list = [PATCHES_DIR + patch.partition('\t')[0] for patch in patch_test_list.readlines()][:8]


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
    patches = []
    d = 1
    for batch in batches:  # for each patch batch
        [x_test, y_test] = batch  # for each patch batch and truth
        preds_train = model.predict(x_test, verbose=0)  # Make Prediction, 9 output channels.
        preds_train_t = (preds_train * 255).astype(np.uint8)  # Scale?
        for i in range(BATCH_SIZE):
            patch_name = next(list_ds_it)
            y_true.append(np.squeeze(y_test[i, 512, 256, :]).argmax())
            y_point_pred.append(preds_train_t[i, 64, 32, :].argmax())
            patch_name = tf.get_static_value(patch_name)
            patches.append(os.path.split(patch_name)[-1].decode('utf-8'))
            print('Patch {} -> Pred {} - Patch {} of {}'.format(patch_name,
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

    results_df = np.array([y_true, y_point_pred, tf.Variable(patches).numpy().tolist()]).T
    column_to_expand = ['Truth', 'Preds', 'Patch']
    dtypes = {'Truth': 'int',
              'Preds': 'int',
              'Patch': 'str'}

    results_df = pd.DataFrame(results_df, columns=column_to_expand).astype(dtype=dtypes)
    results_df['Patch'] = results_df['Patch'].apply(lambda x: x[2:-1].encode().decode("utf-8"))
    results_df[['Institute_id',
                'Patient_no',
                'Image_id',
                'Patch_no']] = results_df['Patch'].str.split('_', expand=True).drop(columns=[4, 5])
    results_df['Patient_id'] = results_df['Institute_id'] + results_df['Patient_no']
    return results_df


with tf.device(DEVICE):
    h5 = load_h5_model(MODEL_PATH)
    results = run_infer(test_list=patch_test_list, model=h5)
    result_file = MODEL_PATH + '_' + str(datetime.now()).replace(':', '.') + '.csv'
    results.to_csv(path_or_buf=result_file, encoding='utf-8')

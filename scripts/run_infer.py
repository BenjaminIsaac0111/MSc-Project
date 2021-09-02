######################################################################
#
# Module: HGDL3.py
# Application: HGDL3 
#
# (c) HeteroGenius Limited 2020
#
######################################################################
#
# Description:
#
# Entry point for HGDL3 deep learning application 
#
######################################################################
#
# Dependencies:
#
# tensorflow
# tensorflow-addons
# numpy
# skimage
# matplotlib (may be removed)
#
######################################################################


# Disable verbose output of tensorflow
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

import tensorflow_addons as tfa

import tensorflow as tf

import tensorflow.keras.losses

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

import logging
from datetime import datetime

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, balanced_accuracy_score, classification_report

import seaborn as sb

from utils import utils
from utils import plotters

tf.autograph.set_verbosity(1)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

# Fool pyinstaller into including code (which is used as a lambda)
GN = tfa.layers.GroupNormalization

# Set some parameters
IMG_WIDTH = 512
IMG_HEIGHT = 1024
IMG_CHANNELS = 3

AUTOTUNE = tf.data.experimental.AUTOTUNE

OUT_CHANNELS = 9


class Linear(tensorflow.keras.layers.Layer):
    def __init__(self, units=32, input_shape=None, name=None, **kwargs):
        super(Linear, self).__init__(kwargs, name=name)
        self.op_shape = input_shape
        self.units = units
        self.b = None

    def build(self, input_shape):
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            # initializer="random_normal",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            trainable=True
        )

    def get_config(self):
        # config={}
        config = super(Linear, self).get_config()
        config['units'] = self.units
        config['input_shape'] = self.op_shape
        return dict(list(config.items()))

    def call(self, inputs):
        return tensorflow.keras.layers.LeakyReLU(alpha=0.01)(
            inputs + tf.broadcast_to(self.b, [self.op_shape[0], self.op_shape[1], self.units]))


def process_path_bit_per_class(file_path, no_channels):
    xLeft, xRight = read_patch(file_path)
    # MIM uses MSB first, then works towards LSB
    bitmask = 128
    c = 0
    output_list = []
    while c <= no_channels:

        masked = tf.bitwise.bitwise_and(tf.cast(bitmask, tf.int32), xRight[:, :, 2]) > 0
        output_list.append(masked)
        c = c + 1

        if c < no_channels:
            masked = tf.bitwise.bitwise_and(tf.cast(bitmask, tf.int32), xRight[:, :, 1]) > 0
            output_list.append(masked)
            c = c + 1

        if c < no_channels:
            masked = tf.bitwise.bitwise_and(tf.cast(bitmask, tf.int32), xRight[:, :, 0]) > 0
            output_list.append(masked)
            c = c + 1

        bitmask = bitmask / 2

    channels = tf.cast(tf.stack(output_list, axis=2), tf.float32)

    return xLeft, channels


def process_path_value_per_class(file_path, no_channels):
    xLeft, xRight = read_patch(file_path)
    c = 1
    output_list = []
    while c <= no_channels:
        masked = xRight[:, :, 2] == c
        output_list.append(masked)
        c = c + 1
    channels = tf.cast(tf.stack(output_list, axis=2), tf.float32)

    return xLeft, channels


def read_patch(file_path):
    img = tf.io.read_file(file_path)
    # NOTE: TF 2.3 docs say is loaded as unit8 - TF 2.2 seem to load a s afloat (check this if upgrading!)
    x = decode_img(img)
    xLeft = tf.slice(x, [0, 0, 0], [1024, 512, 3])
    # xLeft=x
    # Data loaded as float [0,1] range - convert to int [0,255]
    xRight = tf.cast(tf.slice(x, [0, 512, 0], [1024, 512, 3]) * 255, tf.int32)
    return xLeft, xRight


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [1024, 1024])


def prepare_dataset(ds, cache=True, shuffle_buffer_size=1000, prefetch_buffer_size=AUTOTUNE):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    # ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)

    return ds


if __name__ == '__main__':
    component = ['0: Non-Inf',
                 '1: Tumour',
                 '2: Str/Fib',
                 '3: Necr',
                 '4: Vessel',
                 '5: Infl',
                 '6: TumLu',
                 '7: Mucin',
                 '8: Muscle']

    BATCH_SIZE = 14

    dir_path = 'X:/Model_Data/Model_2/'  # Does not matter what model data, all the same in infer.

    no_output_channels = 9

    test_list = open('models/TestData.txt')
    test_list = [dir_path + image.partition('\t')[0] for image in test_list.readlines()]
    # test_list = ['100046_239_Class_1.png',
    #              '100046_23_Class_1.png',
    #              '100046_240_Class_2.png',
    #              '100046_241_Class_1.png',
    #              '100046_242_Class_2.png',
    #              '100046_243_Class_1.png',
    #              '100046_244_Class_6.png',
    #              '100046_245_Class_1.png',
    #              '100046_246_Class_1.png',
    #              '100046_247_Class_1.png',
    #              '100046_248_Class_1.png',
    #              '100046_249_Class_0.png',
    #              ]
    # test_list = [dir_path + image for image in test_list]

    list_ds = tf.data.Dataset.from_tensor_slices(test_list)
    list_ds_it = iter(list_ds)
    num_elements = 1
    for f in list_ds:
        num_elements = num_elements + 1

    labeled_ds = list_ds.map(lambda x: process_path_value_per_class(x, no_output_channels), num_parallel_calls=AUTOTUNE)
    test_ds = prepare_dataset(labeled_ds, cache=None, prefetch_buffer_size=16)

    # TODO Move model to GPU for inference.
    with tf.device('/GPU:0'):
        model = load_model('models/Model_2/DS_MODEL_2_RUN_1.output.h5',
                           compile=False,
                           custom_objects={'LeakyReLU': tensorflow.keras.layers.LeakyReLU(alpha=0.01),
                                           'Linear': Linear})

    batches = iter(test_ds)
    ix = 0

    y_true = []
    y_point_pred = []
    d = 1

    for batch in batches:  # for each patch batch
        [X_train, Y_train] = batch  # for each patch batch and truth
        preds_train = model.predict(X_train, verbose=0)  # Make Prediction, 9 output channels.

        preds_train_t = (preds_train * 255).astype(np.uint8)  # Scale?

        for i in range(BATCH_SIZE):
            # print(np.squeeze(Y_train[i, 512, 256, :]))
            y_true.append(np.squeeze(Y_train[i, 512, 256, :]).argmax())
            # print(preds_train_t[i, 64, 32, :])
            y_point_pred.append(preds_train_t[i, 64, 32, :].argmax())
            print('Patch {} : {} of {} : Pred {}'.format(next(list_ds_it), d, len(test_list), preds_train_t[i, 64, 32, :].argmax()))

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

    print(classification_report(y_true, y_point_pred))
    utils.write_list([str(p) for p in y_point_pred],
                     r'models/Model_2/model_2_round_1_result_' + str(datetime.now()).replace(':', '.') + '.txt')

    #Tumour Cell Density


    # fig = plt.figure(figsize=(10.14 * 2, 5.12 * 2))
    #
    # # Plot First Row
    # fig.add_subplot(2, OUT_CHANNELS + 1, 1)
    # plt.title('Patch Image')
    # plt.imshow(X_train[0])
    #
    # for i in range(0, OUT_CHANNELS):
    #     fig.add_subplot(2, OUT_CHANNELS + 1, i + 2)
    #     plt.title('Class: ' + str(i))
    #     plt.tight_layout()
    #     plt.tick_params(left=False,
    #                     bottom=False,
    #                     labelleft=False,
    #                     labelbottom=False)
    #     plt.imshow(np.squeeze(Y_train[ix][:, :, i]))
    #
    # # Plot Second Row
    # fig.add_subplot(2, OUT_CHANNELS + 1, OUT_CHANNELS + 2)
    # plt.imshow(X_train[0])
    #
    # for i in range(0, OUT_CHANNELS):
    #     fig.add_subplot(2, OUT_CHANNELS + 1, OUT_CHANNELS + 1 + i + 2)
    #     plt.tight_layout()
    #     plt.tick_params(left=False,
    #                     bottom=False,
    #                     labelleft=False,
    #                     labelbottom=False)
    #     plt.imshow(np.squeeze(preds_train_t[ix][:, :, i]))
    #
    # plt.tight_layout()
    # plt.show()
    #
    # if d == 100:
    #     break

    plotters.plot_confusion(y_true=y_true, y_pred=y_point_pred, fmt='d', labels=component)

    # combined_pred = []
    # for p in y_point_pred:
    #     if p != 1 and p != 2 and p != 3:
    #         combined_pred.append(0)
    #     else:
    #         combined_pred.append(p)
    #
    # combined_truth = []
    # for p in y_true:
    #     if p != 1 and p != 2 and p != 3:
    #         combined_truth.append(0)
    #     else:
    #         combined_truth.append(p)
    #
    # cm = {'t': combined_truth, 'y': combined_pred}
    # plotters.plot_confusion(cm)
    # print('Kappa Score: ', cohen_kappa_score(combined_truth, combined_pred))
    # print(classification_report(combined_truth, combined_pred))

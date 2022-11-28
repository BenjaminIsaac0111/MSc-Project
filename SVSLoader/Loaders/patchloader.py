import tensorflow as tf

try:
    AUTOTUNE = tf.data.AUTOTUNE
except AttributeError as e:
    AUTOTUNE = tf.data.experimental.AUTOTUNE


def process_path_value_per_class(file_path=None, no_channels=None, input_size=None):
    xLeft, xRight = load_patch(file_path, input_size)
    c = 1
    output_list = []
    while c <= no_channels:
        masked = xRight[:, :, 2] == c
        output_list.append(masked)
        c = c + 1
    channels = tf.cast(tf.stack(output_list, axis=2), tf.float32)

    return xLeft, channels


def load_patch(file_path=None, input_size=None):
    img = tf.io.read_file(file_path)
    # NOTE: TF 2.3 docs say is loaded as unit8 - TF 2.2 seem to load as a float (check this if upgrading!)
    x = decode_img(img)
    xLeft = tf.slice(x, [0, 0, 0], [input_size[0], input_size[1], 3])
    # xLeft=x
    # Data loaded as float [0,1] range - convert to int [0,255]
    xRight = tf.cast(tf.slice(x, [0, input_size[1], 0], [input_size[0], input_size[1], 3]) * 255, tf.int32)
    return xLeft, xRight


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img


def prepare_dataset(ds, batch_size=16, cache=True, shuffle=False, shuffle_buffer_size=128,
                    prefetch_buffer_size=AUTOTUNE, repeat=None):
    # If this is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=prefetch_buffer_size)
    # Repeat forever
    ds = ds.repeat(repeat)
    return ds

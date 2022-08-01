from tensorflow import keras
from tensorflow.keras import layers


def xception_unet(input_size=(256, 256), num_classes=9, filters=None, n_conv_layers=2):
    if filters is None:
        filters = [64, 128, 256]
    skip_connections = []

    inputs = keras.Input(shape=input_size + (3,))
    s = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    # Entry block
    x = layers.SeparableConv2D(64, 2, strides=1, padding="same", activation='leaky_relu',
                               kernel_initializer='HeNormal')(s)
    x = layers.BatchNormalization()(x)
    residual_skip_connection = x
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for f in filters:
        for _ in range(n_conv_layers):
            x = layers.SeparableConv2D(f, 2, padding="same", activation='leaky_relu', kernel_initializer='HeNormal')(x)
            x = layers.BatchNormalization()(x)
        skip_connections.append(x)  # store for context skips.
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        residual = layers.Conv2D(f, 1, strides=2, activation='leaky_relu', padding="same")(residual_skip_connection)
        x = layers.add([x, residual])  # Add back residual
        residual_skip_connection = x

    x = layers.Conv2D(f * 2, 2, padding="same", activation='leaky_relu', kernel_initializer='HeNormal')(x)

    filters.reverse()

    for f in filters:
        for _ in range(n_conv_layers):
            x = layers.Conv2D(f, 2, padding="same", activation='leaky_relu', kernel_initializer='HeNormal')(x)
            x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(f, 2, strides=2, padding="same", activation='leaky_relu',
                                   kernel_initializer='HeNormal')(x)
        residual = layers.Conv2DTranspose(f, 2, strides=2, padding="same", activation='leaky_relu',
                                          kernel_initializer='HeNormal')(residual_skip_connection)
        x = layers.add([x, residual])  # Add back residual
        if len(skip_connections) > 0:
            x = layers.concatenate([x, skip_connections.pop()])
        residual_skip_connection = x
    # Add a per-pixel classification layer
    fully_connected_network = layers.Conv2D(num_classes, 2, activation="softmax", padding="same",
                                            kernel_initializer='HeNormal')(x)
    return inputs, fully_connected_network


def res_unet(filter_root=64, depth=4, n_class=9, input_size=(256, 256, 3), activation='leaky_relu', batch_norm=True,
             final_activation='softmax'):
    inputs = layers.Input(input_size)
    x = inputs
    # Dictionary for long connections
    long_connection_store = {}

    # Down sampling
    for i in range(depth):
        out_channel = 2 ** i * filter_root

        # Residual/Skip connection
        res = layers.Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False)(x)

        # First Conv Block with Conv, BN and activation
        conv1 = layers.Conv2D(out_channel, kernel_size=3, padding='same')(x)
        if batch_norm:
            conv1 = layers.BatchNormalization()(conv1)
        act1 = layers.Activation(activation)(conv1)

        # Second Conv block with Conv and BN only
        conv2 = layers.Conv2D(out_channel, kernel_size=3, padding='same')(act1)
        if batch_norm:
            conv2 = layers.BatchNormalization()(conv2)

        resconnection = layers.Add()([res, conv2])

        act2 = layers.Activation(activation)(resconnection)

        # Max pooling
        if i < depth - 1:
            long_connection_store[str(i)] = act2
            x = layers.MaxPooling2D(padding='same')(act2)
        else:
            x = act2

    # Upsampling
    for i in range(depth - 2, -1, -1):
        out_channel = 2 ** i * filter_root

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]

        up_conv1 = layers.Conv2DTranspose(out_channel, 2, activation=activation, padding='same')(x)

        #  Concatenate.
        up_conc = layers.Concatenate(axis=-1)([up_conv1, long_connection])

        #  Convolutions
        up_conv2 = layers.Conv2D(out_channel, 3, padding='same')(up_conc)
        if batch_norm:
            up_conv2 = layers.BatchNormalization()(up_conv2)
        up_act1 = layers.Activation(activation)(up_conv2)

        up_conv2 = layers.Conv2D(out_channel, 3, padding='same')(up_act1)
        if batch_norm:
            up_conv2 = layers.BatchNormalization()(up_conv2)

        # Residual/Skip connection
        res = layers.Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False)(up_conc)

        resconnection = layers.Add()([res, up_conv2])

        x = layers.Activation(activation)(resconnection)

    # Final convolution
    output = layers.Conv2D(n_class, 1, padding='same', activation=final_activation, name='output')(x)

    return inputs, output


def unet(input_shape=(128, 128, 3), n_labels=9, n_filters=32, output_mode="softmax"):
    inputs = layers.Input(input_shape)

    conv1 = layers.Convolution2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation("leaky_relu")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Convolution2D(2 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation("leaky_relu")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Convolution2D(4 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation("leaky_relu")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Convolution2D(8 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation("leaky_relu")(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Convolution2D(16 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("leaky_relu")(conv5)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    conv6 = layers.Convolution2D(8 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(up6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Activation("leaky_relu")(conv6)
    merge6 = layers.concatenate([conv4, conv6], axis=3)

    up7 = layers.UpSampling2D(size=(2, 2))(merge6)
    conv7 = layers.Convolution2D(4 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(up7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Activation("leaky_relu")(conv7)
    merge7 = layers.concatenate([conv3, conv7], axis=3)

    up8 = layers.UpSampling2D(size=(2, 2))(merge7)
    conv8 = layers.Convolution2D(2 * n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(up8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Activation("leaky_relu")(conv8)
    merge8 = layers.concatenate([conv2, conv8], axis=3)

    up9 = layers.UpSampling2D(size=(2, 2))(merge8)
    conv9 = layers.Convolution2D(n_filters, (3, 3), padding='same', kernel_initializer='he_normal')(up9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Activation("leaky_relu")(conv9)
    merge9 = layers.concatenate([conv1, conv9], axis=3)

    conv10 = layers.Convolution2D(n_labels, (1, 1), padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = layers.BatchNormalization()(conv10)
    outputs = layers.Activation(output_mode)(conv10)

    return inputs, outputs


def inception_module(inputs, numFilters=32):
    tower_0 = layers.Conv2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    tower_0 = layers.BatchNormalization()(tower_0)
    tower_0 = layers.Activation("leaky_relu")(tower_0)

    tower_1 = layers.Conv2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    tower_1 = layers.BatchNormalization()(tower_1)
    tower_1 = layers.Activation("leaky_relu")(tower_1)
    tower_1 = layers.Conv2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_1)
    tower_1 = layers.BatchNormalization()(tower_1)
    tower_1 = layers.Activation("leaky_relu")(tower_1)

    tower_2 = layers.Conv2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    tower_2 = layers.BatchNormalization()(tower_2)
    tower_2 = layers.Activation("leaky_relu")(tower_2)
    tower_2 = layers.Conv2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_2)
    tower_2 = layers.Conv2D(numFilters, (3, 3), padding='same', kernel_initializer='he_normal')(tower_2)
    tower_2 = layers.BatchNormalization()(tower_2)
    tower_2 = layers.Activation("leaky_relu")(tower_2)

    tower_3 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    tower_3 = layers.Conv2D(numFilters, (1, 1), padding='same', kernel_initializer='he_normal')(tower_3)
    tower_3 = layers.BatchNormalization()(tower_3)
    tower_3 = layers.Activation("leaky_relu")(tower_3)

    inception_module = layers.concatenate([tower_0, tower_1, tower_2, tower_3], axis=3)
    return inception_module


def inception_unet(input_shape=(128, 128, 3), n_labels=9, n_filters=32, output_mode="softmax"):
    inputs = layers.Input(input_shape)

    conv1 = inception_module(inputs, n_filters)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = inception_module(pool1, 2 * n_filters)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = inception_module(pool2, 4 * n_filters)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = inception_module(pool3, 8 * n_filters)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = inception_module(pool4, 16 * n_filters)

    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = inception_module(up6, 8 * n_filters)
    merge6 = layers.concatenate([conv4, up6], axis=3)

    up7 = layers.UpSampling2D(size=(2, 2))(merge6)
    up7 = inception_module(up7, 4 * n_filters)
    merge7 = layers.concatenate([conv3, up7], axis=3)

    up8 = layers.UpSampling2D(size=(2, 2))(merge7)
    up8 = inception_module(up8, 2 * n_filters)
    merge8 = layers.concatenate([conv2, up8], axis=3)

    up9 = layers.UpSampling2D(size=(2, 2))(merge8)
    up9 = inception_module(up9, n_filters)
    merge9 = layers.concatenate([conv1, up9], axis=3)

    conv10 = layers.Convolution2D(n_labels, (1, 1), padding='same', kernel_initializer='he_normal')(merge9)
    conv10 = layers.BatchNormalization()(conv10)
    outputs = layers.Activation(output_mode)(conv10)

    return inputs, outputs


if __name__ == '__main__':
    # Free up RAM in case the ssl_model definition cells were run multiple times
    keras.backend.clear_session()
    inputs, fc_output = res_unet()
    model = keras.Model(inputs, fc_output)
    model.summary()

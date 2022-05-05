import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def encoder_mini_block(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the
    decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and projector_width (hence,
    # is not reduced in size)
    conv = layers.Conv2D(n_filters,
                         3,  # Kernel size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(inputs)
    conv = layers.Conv2D(n_filters,
                         3,  # Kernel size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = layers.BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of
    # weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder
    # block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to
    # traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during
    # transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


def decoder_mini_block(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = layers.Conv2DTranspose(
        n_filters,
        (3, 3),  # Kernel size
        strides=(2, 2),
        padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = layers.concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = layers.Conv2D(n_filters,
                         3,  # Kernel size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(merge)
    conv = layers.Conv2D(n_filters,
                         3,  # Kernel size
                         activation='relu',
                         padding='same',
                         kernel_initializer='HeNormal')(conv)
    return conv


def u_net_compiled(input_size=(512, 512, 3), n_filters=32, n_classes=3):
    """
    Combine both encoder and decoder blocks according to the U-Net research paper
    Return the ssl_model as output
    """
    # Input size represent the size of 1 image (the size used for pre-processing)
    inputs = layers.Input(input_size)

    # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
    # Observe that the filter are increasing as we go deeper into the network which will increasse the # channels of
    # the image
    cblock1 = encoder_mini_block(inputs, n_filters, dropout_prob=0, max_pooling=True)
    cblock2 = encoder_mini_block(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
    cblock3 = encoder_mini_block(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
    cblock4 = encoder_mini_block(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True)
    cblock5 = encoder_mini_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

    # Decoder includes multiple mini blocks with decreasing number of filter
    # Observe the skip connections from the encoder are given as input to the decoder
    # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
    ublock6 = decoder_mini_block(cblock5[0], cblock4[1], n_filters * 8)
    ublock7 = decoder_mini_block(ublock6, cblock3[1], n_filters * 4)
    ublock8 = decoder_mini_block(ublock7, cblock2[1], n_filters * 2)
    ublock9 = decoder_mini_block(ublock8, cblock1[1], n_filters)

    # Complete the ssl_model with 1 3x3 convolution layer (Same as the prev Conv Layers)
    # Followed by a 1x1 Conv layer to get the image to the desired size.
    # Observe the number of channels will be equal to number of output classes
    conv9 = layers.Conv2D(n_filters,
                          3,
                          activation='relu',
                          padding='same',
                          kernel_initializer='he_normal')(ublock9)

    conv10 = layers.Conv2D(n_classes, 1, padding='same')(conv9)

    # Define the ssl_model
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


def xception_unet(img_size=(512, 512), num_classes=9, filters=None, n_conv_layers=2):
    if filters is None:
        filters = [64, 128, 256]

    inputs = keras.Input(shape=img_size + (3,))
    s = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    # Entry block
    x = layers.Conv2D(64, 3, strides=1, padding="same", activation='relu', kernel_initializer='HeNormal')(s)
    x = layers.BatchNormalization()(x)
    previous_block_activation = x  # Set aside residual
    # TODO make n Convs variable.
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for f in filters:
        x = layers.SeparableConv2D(f, 3, padding="same", activation='relu', kernel_initializer='HeNormal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.SeparableConv2D(f, 3, padding="same", activation='relu', kernel_initializer='HeNormal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
        # Project residual
        residual = layers.Conv2D(f, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    enc = x

    filters.reverse()
    for f in filters:
        x = layers.Conv2DTranspose(f, 3, padding="same", activation='relu', kernel_initializer='HeNormal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2DTranspose(f, 3, padding="same", activation='relu', kernel_initializer='HeNormal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(2)(x)
        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(f, 1, padding="same", kernel_initializer='HeNormal')(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    # Add a per-pixel classification layer
    dec = layers.Conv2D(num_classes, 3, activation="softmax", padding="same", kernel_initializer='HeNormal')(x)
    # Define the ssl_model
    return inputs, enc, dec


if __name__ == '__main__':
    # Free up RAM in case the ssl_model definition cells were run multiple times
    keras.backend.clear_session()
    # Build ssl_model
    inputs, encoder, decoder = xception_unet(filters=[64, 128, 256, 512])
    encoder = keras.Model(inputs, encoder)

    decoder = keras.Model(inputs, decoder)
    # Transfer Weights from encoder
    for i, layer in enumerate(encoder.layers):
        print(layer)
        decoder.layers[i].set_weights(layer.get_weights())
        print(i)

    # skips_idx = [(i, ssl_model.get_layer(layer.name))
    #              for i, layer in enumerate(ssl_model.layers)
    #              if 'add' in layer.name][:-1]
    # ssl_model.summary()
    # encoder = keras.Model(ssl_model.layers[:30], ssl_model.layers[:30].output)
    # keras.utils.plot_model(ssl_model, to_file='ssl_model.png', show_layer_activations=True, show_shapes=True, rankdir='LR')
    # keras.utils.plot_model(encoder, to_file='encoder.png', show_layer_activations=True, show_shapes=True, rankdir='LR')
    # keras.utils.plot_model(ssl_model, to_file='decoder.png', show_layer_activations=True, show_shapes=True, rankdir='LR')

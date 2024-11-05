#!/usr/bin/env python3
"""Inception Block Module"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Function that builds an inception block as described in Going
    Deeper with Convolutions (2014):

    A_prev = the output from the previous layer
    filters = a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
    respectively:
    F1 = the number of filters in the 1x1 convolution
    F3R = the number of filters in the 1x1 convolution before the 3x3
    convolution
    F3 = the number of filters in the 3x3 convolution
    F5R = the number of filters in the 1x1 convolution before the 5x5
    convolution
    F5 = the number of filters in the 5x5 convolution
    FPP = the number of filters in the 1x1 convolution after the max
    pooling
    All convolutions inside the inception block should use a rectified
    linear activation (ReLU)
    Returns: the concatenated output of the inception block

    """
    # unpack filter tuples
    F1, F3R, F3, F5R, F5, FPP = filters

    # F1 convolution
    conv1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                            activation='relu')(A_prev)

    # F1 then followed by F3
    conv3 = K.layers.Conv2D(F3R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            activation='relu')(conv3)

    # F1 then followed by F5
    conv5 = K.layers.Conv2D(F5R, (1, 1), padding='same',
                            activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                            activation='relu')(conv5)

    # pooling the layers
    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                 padding='same')(A_prev)
    pool = K.layers.Conv2D(FPP, (1, 1), padding='same',
                           activation='relu')(pool)

    # concatenating
    conc_output = K.layers.Concatenate(axis=-1)([conv1, conv3, conv5, pool])

    # returns the concatenate output of the inception block
    return (conc_output)
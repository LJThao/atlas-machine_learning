#!/usr/bin/env python3
"""Inception Network Module"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function that builds the inception network as described in Going
    Deeper with Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use a
    rectified linear activation (ReLU)
    You may use inception_block = __import__('0-inception_block').
    inception_block
    Returns: the keras model

    """
    # set input to the shape
    input = K.Input(shape=(224, 224, 3))

    # init the layers
    x = K.layers.Conv2D(64, (7, 7), strides=2,
                        padding='same', activation='relu')(input)
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    x = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # incept block filter tuples as (F1, F3R, F3. F5R, F5, FPP)
    filters = [
        (64, 96, 128, 16, 32, 32), (128, 128, 192, 32, 96, 64),
        (192, 96, 208, 16, 48, 64), (160, 112, 224, 24, 64, 64),
        (128, 128, 256, 24, 64, 64), (112, 144, 288, 32, 64, 64),
        (256, 160, 320, 32, 128, 128), (256, 160, 320, 32, 128, 128),
        (384, 192, 384, 48, 128, 128)
    ]

    # apply incept blocks and adding the pooling layers
    for i, f in enumerate(filters):
        x = inception_block(x, f)
        if i == 1 or i == 6:
            x = K.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # set x to the finished layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dropout(0.4)(x)
    output = K.layers.Dense(1000, activation='softmax')(x)

    # set model
    keras_model = K.Model(input, output)

    # returns the keras model
    return (keras_model)

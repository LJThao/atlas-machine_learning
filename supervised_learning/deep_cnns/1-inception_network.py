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
    Returns: the keras model

    """
    # set input to the shape
    input = K.Input(shape=(224, 224, 3))

    # init the layers
    x = K.layers.Conv2D(64, (7, 7), strides=2,
                        padding='same', activation='relu')(input)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    x = K.layers.Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = K.layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # incept layers
    incept_3a = inception_block(x, [64, 96, 128, 16, 32, 32])
    incept_3b = inception_block(incept_3a, [128, 128, 192, 32, 96, 64])
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(incept_3b)

    incept_4a = inception_block(x, [192, 96, 208, 16, 48, 64])
    incept_4b = inception_block(incept_4a, [160, 112, 224, 24, 64, 64])
    incept_4c = inception_block(incept_4b, [128, 128, 256, 24, 64, 64])
    incept_4d = inception_block(incept_4c, [112, 144, 288, 32, 64, 64])
    incept_4e = inception_block(incept_4d, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(incept_4e)

    incept_5a = inception_block(x, [256, 160, 320, 32, 128, 128])
    incept_5b = inception_block(incept_5a, [384, 192, 384, 48, 128, 128])

    # set x to the finished layers
    x = K.layers.GlobalAveragePooling2D()(incept_5b)
    x = K.layers.Dropout(0.4)(x)
    output = K.layers.Dense(1000, activation='softmax')(x)

    # set model
    keras_model = K.Model(input, output)

    # returns the keras model
    return (keras_model)

#!/usr/bin/env python3
"""ResNet-50 Module"""
from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture as described
    in Deep Residual Learning for Image Recognition (2015):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by
    batch normalization along the channels axis and a rectified linear
    activation (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the keras model

    """
    # init the HeNormal and input for shape
    init = K.initializers.HeNormal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    # init conv and pooling
    layer_output = K.layers.Conv2D(64, 7,
                                   strides=2,
                                   padding='same',
                                   kernel_initializer=init
                                   )(inputs)
    layer_output = K.layers.BatchNormalization(axis=3)(layer_output)
    layer_output = K.layers.Activation('relu')(layer_output)
    layer_output = K.layers.MaxPooling2D(3,
                                         strides=2,
                                         padding='same'
                                         )(layer_output)

    # the ResNet Stages
    stages = [
        ([64, 64, 256], 3, 1),
        ([128, 128, 512], 4, 2),
        ([256, 256, 1024], 6, 2),
        ([512, 512, 2048], 3, 2)
    ]
    # iterate over the stages
    for filters, blocks, stride in stages:
        layer_output = projection_block(layer_output, filters, s=stride)
        for _ in range(1, blocks):
            layer_output = identity_block(layer_output, filters)

    # average the pooling and connect the layer
    layer_output = K.layers.GlobalAveragePooling2D()(layer_output)
    outputs = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(layer_output)
    model = K.Model(inputs=inputs, outputs=outputs)

    # returns the keras model 
    return model

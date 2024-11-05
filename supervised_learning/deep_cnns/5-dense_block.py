#!/usr/bin/env python3
"""Dense Block Module"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Function that builds a dense block as described in Densely
    Connected Convolutional Networks:

    X = the output from the previous layer
    nb_filters = an integer representing the number of filters in X
    growth_rate = the growth rate for the dense block
    layers = the number of layers in the dense block
    You should use the bottleneck layers used for DenseNet-B
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    All convolutions should be preceded by Batch Normalization and a
    rectified linear activation (ReLU), respectively
    Returns: The concatenated output of each layer within the Dense Block
    and the number of filters within the concatenated outputs, respectively

    """

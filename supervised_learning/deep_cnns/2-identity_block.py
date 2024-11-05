#!/usr/bin/env python3
"""Identity Block Module"""
from tensorflow import keras as K


def identity_block(A_prev, filters): 
    """Function that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015):

    A_prev = the output from the previous layer
    filters = a tuple or list containing F11, F3, F12, respectively:
    F11 = the number of filters in the first 1x1 convolution
    F3 = the number of filters in the 3x3 convolution
    F12 = the number of filters in the second 1x1 convolution
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the identity block

    """

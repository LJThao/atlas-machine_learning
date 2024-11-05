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

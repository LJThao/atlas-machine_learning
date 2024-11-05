#!/usr/bin/env python3
"""Projection Block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """Function that builds a projection block as described in
    Deep Residual Learning for Image Recognition (2015):

    A_prev = the output from the previous layer
    filters = a tuple or list containing F11, F3, F12, respectively:
    F11 = the number of filters in the first 1x1 convolution
    F3 = the number of filters in the 3x3 convolution
    F12 = the number of filters in the second 1x1 convolution as well
    as the 1x1 convolution in the shortcut connection
    s = the stride of the first convolution in both the main path and
    the shortcut connection
    All convolutions inside the block should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    The seed for the he_normal initializer should be set to zero
    Returns: the activated output of the projection block

    """

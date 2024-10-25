#!/usr/bin/env python3
"""Convolutional Forward Prop Module"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Function that performs forward propagation over a convolutional layer
    of a neural network:

    A_prev = a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
    m = the number of examples
    h_prev = the height of the previous layer
    w_prev = the width of the previous layer
    c_prev = the number of channels in the previous layer
    W = a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
    kh = the filter height
    kw = the filter width
    c_prev = the number of channels in the previous layer
    c_new = the number of channels in the output
    b = a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation = an activation function applied to the convolution
    padding = a string that is either same or valid, indicating the type
    of padding used
    stride = a tuple of (sh, sw) containing the strides for the convolution
    sh = the stride for the height
    sw = the stride for the width
    * only import numpy as np *
    Returns: the output of the convolutional layer

    """

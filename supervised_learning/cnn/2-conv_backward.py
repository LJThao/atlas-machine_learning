#!/usr/bin/env python3
"""Convolutional Back Prop Module"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Function that performs back propagation over a convolutional layer
    of a neural network:

    dZ = a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the unactivated output of the
    convolutional layer
    m = the number of examples
    h_new = the height of the output
    w_new = the width of the output
    c_new = the number of channels in the output
    A_prev = a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
    h_prev = the height of the previous layer
    w_prev = the width of the previous layer
    c_prev = the number of channels in the previous layer
    W = a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
    kernels for the convolution
    kh = the filter height
    kw = the filter width
    b = a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding = a string that is either same or valid, indicating the type of
    padding used
    stride = a tuple of (sh, sw) containing the strides for the convolution
    sh = the stride for the height
    sw = the stride for the width
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev), the kernels (dW), and the biases (db), respectively

    """

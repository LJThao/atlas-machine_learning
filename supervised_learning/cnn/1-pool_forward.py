#!/usr/bin/env python3
"""Pooling Forward Prop Module"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs forward propagation over a pooling layer of a
    neural network:

    A_prev = a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
    the output of the previous layer
    m = the number of examples
    h_prev = the height of the previous layer
    w_prev = the width of the previous layer
    c_prev = the number of channels in the previous layer
    kernel_shape = a tuple of (kh, kw) containing the size of the kernel for
    the pooling
    kh = the kernel height
    kw = the kernel width
    stride = a tuple of (sh, sw) containing the strides for the pooling
    sh = the stride for the height
    sw = the stride for the width
    mode = a string containing either max or avg, indicating whether to perform
    maximum or average pooling, respectively
    you may import numpy as np
    Returns: the output of the pooling layer

    """

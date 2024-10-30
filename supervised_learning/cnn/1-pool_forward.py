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
    # unpack
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride

    # calculate the outputs of h and w
    h_new = (h_prev - kh) // sh + 1
    w_new = (w_prev - kw) // sw + 1

    # init the output
    output_mat = np.zeros((m, h_new, w_new, c_prev))

    # apply the pooling operation
    for y in range(h_new):
        for x in range(w_new):
            y_start = y * sh
            y_end = y_start + kh
            x_start = x * sw
            x_end = x_start + kw

            # extracts the region of the input for pooling
            region = A_prev[:, y_start:y_end, x_start:x_end, :]

            # applying max and avg to each region
            if mode == 'max':
                output_mat[:, y, x, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output_mat[:, y, x, :] = np.mean(region, axis=(1, 2))

    # returns output of the pooling layer
    return (output_mat)
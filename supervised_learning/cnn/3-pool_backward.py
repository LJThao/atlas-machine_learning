#!/usr/bin/env python3
"""Pooling Back Prop Module"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Function that performs back propagation over a pooling layer of a
    neural network:

    dA = a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
    partial derivatives with respect to the output of the pooling layer
    m = the number of examples
    h_new = the height of the output
    w_new = the width of the output
    c = the number of channels
    A_prev = a numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
    output of the previous layer
    h_prev = the height of the previous layer
    w_prev = the width of the previous layer
    kernel_shape = a tuple of (kh, kw) containing the size of the kernel for
    the pooling
    kh = the kernel height
    kw = the kernel width
    stride = a tuple of (sh, sw) containing the strides for the pooling
    sh = the stride for the height
    sw = the stride for the width
    mode = a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    you may import numpy as np
    Returns: the partial derivatives with respect to the previous layer
    (dA_prev)

    """
    # unpack
    (m, h_prev, w_prev, c) = A_prev.shape
    (kh, kw) = kernel_shape
    (sh, sw) = stride
    (m, h_new, w_new, c_new) = dA.shape

    # init dA_prev
    dA_prev = np.zeros_like(A_prev)

    # iterates over each training examples, height, width
    for i in range(m):
        for y in range(h_new):
            for x in range(w_new):
                # iterates over c and apply backprop for the pooling op
                for c_ in range(c):
                    # calculates the start and end positions
                    y_start = y * sh
                    y_end = y_start + kh
                    x_start = x * sw
                    x_end = x_start + kw

                    # apply backprop for max pooling
                    if mode == 'max':
                        pool_slice = A_prev[
                            i,
                            y_start:y_end,
                            x_start:x_end,
                            c_
                        ]
                        mask = (pool_slice == np.max(pool_slice))
                        dA_prev[i,
                                y_start:y_end, x_start:x_end,
                                c_] += mask * dA[i, y, x, c_]

                    # apply backprop for avg pooling
                    elif mode == 'avg':
                        da = dA[i, y, x, c_]
                        average = da / (kh * kw)
                        dA_prev[i,
                                y_start:y_end,
                                x_start:x_end,
                                c_] += np.ones((kh, kw)) * average

    # returns the partial derivatives with respect to the previous layer
    return (dA_prev)

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
    # unpack
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, _, c_new) = W.shape
    (sh, sw) = stride

    # determining the padding
    if padding == "valid":
        (pad_y, pad_x) = 0, 0
    elif padding == "same":
        pad_y = (((h_prev - 1) * sh) + kh - h_prev) // 2
        pad_x = (((w_prev - 1) * sw) + kw - w_prev) // 2

    # padding the inputs
    A_padded = np.pad(
        A_prev,
        ((0, 0),
         (pad_y, pad_y),
         (pad_x, pad_x),
         (0, 0))
    )

    # calculate y and x outputs
    y_output = ((h_prev + 2 * pad_y - kh) // sh) + 1
    x_output = ((w_prev + 2 * pad_x - kw) // sw) + 1

    # init the output
    output_mat = np.zeros((m, y_output, x_output, c_new))

    # apply convolution operation
    for y in range(y_output):
        for x in range(x_output):
            y_start = y * sh
            y_end = y_start + kh
            x_start = x * sw
            x_end = x_start + kw

            # apply tensordot function
            input_slice = A_padded[:, y_start:y_end, x_start:x_end, :]
            conv_output = np.tensordot(
                input_slice, W, axes=([1, 2, 3], [0, 1, 2]))
            output_mat[:, y, x, :] = conv_output

    # apply activation then returns the output
    return activation(output_mat + b)
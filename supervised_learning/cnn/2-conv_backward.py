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
    # unpack dZ, A_prev, W
    (m, h_new, w_new, c_new) = dZ.shape
    (m, h_prev, w_prev, c_prev) = A_prev.shape
    (kh, kw, _, _) = W.shape
    (sh, sw) = stride

    # init dA_prev, dW, db
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # determining padding
    if padding == "same":
        pad_h = (((h_prev - 1) * sh + kh - h_new) // 2)
        pad_w = (((w_prev - 1) * sw + kw - w_new) // 2)
    else:
        pad_h, pad_w = 0, 0

    # pad the A_prev and dA_prev for backpropagation
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0),
         (pad_h, pad_h),
         (pad_w, pad_w),
         (0, 0))
    )
    dA_prev_pad = np.pad(
        dA_prev,
        ((0, 0),
         (pad_h, pad_h),
         (pad_w, pad_w),
         (0, 0))
    )

    # apply backpropagation over the convolution
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]

        for y in range(h_new):
            for x in range(w_new):
                for c in range(c_new):

                    # determining the slice coordinates
                    y_start = y * sh
                    y_end = y_start + kh
                    x_start = x * sw
                    x_end = x_start + kw

                    # slice the A_prev and update gradients
                    a_slice = a_prev_pad[y_start:y_end, x_start:x_end, :]

                    # update gradients for the window
                    da_prev_pad[
                        y_start:y_end,
                        x_start:x_end, :] += W[:, :, :, c] * dZ[i, y, x, c]
                    dW[:, :, :, c] += a_slice * dZ[i, y, x, c]

        # unpadding the dA_prev
        if padding == "same":
            dA_prev[i, :, :, :] = da_prev_pad[pad_h:-pad_h, pad_w:-pad_w, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    # returns the previous layer, kernels, and biases
    return (dA_prev, dW, db)

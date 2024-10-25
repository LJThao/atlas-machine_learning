#!/usr/bin/env python3
"""Pool(images, kernel_shape, stride, mode='max'): Module
-- Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Function that performs pooling on images:

    images = a numpy.ndarray with shape (m, h, w, c) containing multiple
    images
    m = the number of images
    h = the height in pixels of the images
    w = the width in pixels of the images
    c = the number of channels in the image
    kernel_shape = a tuple of (kh, kw) containing the kernel shape for the
    pooling
    kh = the height of the kernel
    kw = the width of the kernel
    stride = a tuple of (sh, sw)
    sh = the stride for the height of the image
    sw = the stride for the width of the image
    mode indicates the type of pooling
    max indicates max pooling
    avg indicates average pooling
    You are only allowed to use two for loops; any other loops of any kind
    are not allowed
    Returns: a numpy.ndarray containing the pooled images

    """
    # unpack images, kernel, stride
    (m, h, w, c), (kh, kw), (sh, sw) = images.shape, kernel_shape, stride

    # calculate output for height and width
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # init the output
    pooled_images = np.zeros((m, output_h, output_w, c))

    # apply pooling
    for y in range(output_h):
        for x in range(output_w):
            # retrieve the current stride region and kernel size
            current_stride = images[
                :,
                y * sh:y * sh + kh,
                x * sw:x * sw + kw,
                :
            ]

            # applying the operation for pooling (max and avg)
            if mode == 'max':
                pooled_images[:, y, x, :] = np.max(
                    current_stride, axis=(1, 2))
            elif mode == 'avg':
                pooled_images[:, y, x, :] = np.mean(
                    current_stride, axis=(1, 2))

    # returns a numpy.ndarray containing the pooled images
    return (pooled_images)

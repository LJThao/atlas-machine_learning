#!/usr/bin/env python3
"""Convolve_Grayscale_Same(images, kernel) Module
-- Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images:

    images = a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    m = the number of images
    h = the height in pixels of the images
    w = the width in pixels of the images
    kernel = a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
    kh = the height of the kernel
    kw = the width of the kernel
    if necessary, the image should be padded with 0's
    You are only allowed to use two for loops; any other loops of any
    kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    # unpack images and kernel
    (m, h, w), (kh, kw) =  images.shape, kernel.shape

    # calculate the padding size
    pad_h1 = (kh - 1) // 2
    pad_w1 = (kw - 1) // 2
    pad_h2 = (kh - 1) % 2
    pad_w2 = (kw - 1) % 2

    # pad the images with zeros
    padded_images = np.pad(images, ((0, 0),
                                    (pad_h1, pad_h1 + pad_h2),
                                    (pad_w1, pad_w1 + pad_w2))
    )

    # init the output
    convolved_images = np.zeros((m, h, w))

    # apply convolution using tensordot function
    for y in range(h):
        for x in range(w):
            convolved_images[:, y, x] = np.tensordot(
                padded_images[:, y:y + kh, x:x + kw],
                kernel, axes=((1, 2), (0, 1))
            )

    # return a numpy.ndarry containing the convolved images
    return (convolved_images)

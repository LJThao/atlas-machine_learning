#!/usr/bin/env python3
"""Convolve_Grayscale_Valid(images, kernel) Module"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Function that performs a valid convolution on grayscale images:

    images = a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
    m = the number of images
    h = the height in pixels of the images
    w = the width in pixels of the images
    kernel = a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
    kh = the height of the kernel
    kw = the width of the kernel
    You are only allowed to use two for loops; any other loops of any kind
    are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    # calculate the output
    m = images.shape[0]
    kh, kw = kernel.shape
    output_h, output_w = [
        m_size - k_size + 1
        for m_size, k_size in zip(images.shape[1:], kernel.shape)
    ]

    # init the output with zeros
    convolved_images = np.zeros((m, output_h, output_w))

    # apply convolution
    for y in range(output_h):
        for x in range(output_w):
            # apply tensordot function
            convolved_images[:, y, x] = np.tensordot(
                images[:, y:y + kh, x:x + kw],
                kernel, axes=((1, 2), (0, 1)))

    # return a numpy.ndarry containing the convolved images
    return (convolved_images)

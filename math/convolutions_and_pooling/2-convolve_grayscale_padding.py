#!/usr/bin/env python3
"""Convolve_Grayscale_Padding(images, kernel, padding) Module
-- Padding Convolution"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Function that performs a convolution on grayscale
    images with custom padding:

    images = a numpy.ndarray with shape (m, h, w) containing
    multiple grayscale images
    m = the number of images
    h = the height in pixels of the images
    w = the width in pixels of the images
    kernel = a numpy.ndarray with shape (kh, kw) containing the
    kernel for the convolution
    kh = the height of the kernel
    kw = the width of the kernel
    padding = a tuple of (ph, pw)
    ph = the padding for the height of the image
    pw = the padding for the width of the image
    the image should be padded with 0's
    You are only allowed to use two for loops; any other loops of
    any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    # unpack images, kernel, padding
    (m, h, w), (kh, kw), (pad_h, pad_w) = images.shape, kernel.shape, padding

    # pad the images with zeros with the padding values
    padded_images = np.pad(
        images,
        ((0, 0),
         (pad_h, pad_h),
         (pad_w, pad_w))
    )

    # calculate the output height and width
    output_h = h + 2 * pad_h - kh + 1
    output_w = w + 2 * pad_w - kw + 1

    # init the output
    convolved_images = np.zeros((m, output_h, output_w))

    # apply convolution using tensordot function
    for y in range(output_h):
        for x in range(output_w):
            convolved_images[:, y, x] = np.tensordot(
                padded_images[:, y:y + kh, x:x + kw],
                kernel, axes=((1, 2), (0, 1))
            )

    # returns the numpy.ndarray containing the convolved images
    return (convolved_images)

#!/usr/bin/env python3
"""Convolve_Grayscale_Same(images, kernel) Module
-- Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Function that performs a same convolution on grayscale images:

    images = a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
    m = the number of images
    h = the height in pixels of the images
    w = the width in pixels of the images
    kernel = a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution
    kh = the height of the kernel
    kw = the width of the kernel
    if necessary, the image should be padded with 0's
    You are only allowed to use two for loops; any other loops of any kind are not allowed
    Returns: a numpy.ndarray containing the convolved images

    """
    
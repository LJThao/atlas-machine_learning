#!/usr/bin/env python3
"""Function that concatenates two matrices"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Return a new numpy.ndarray"""
    return np.concatenate((mat1, mat2), axis=axis)

#!/usr/bin/env python3
"""Correlation (C) Module"""
import numpy as np


def correlation(C):
    """Function that calculates a correlation matrix:

    C is a numpy.ndarray of shape (d, d) containing a covariance
    matrix
    d is the number of dimensions
    If C is not a numpy.ndarray, raise a TypeError with the
    message C must be a numpy.ndarray
    If C does not have shape (d, d), raise a ValueError with
    the message C must be a 2D square matrix
    Returns a numpy.ndarray of shape (d, d) containing the
    correlation matrix

    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # computing the std
    std_dev = np.sqrt(np.diag(C))

    # compute the correlation matrix
    correlation_mat = C / np.outer(std_dev, std_dev)

    # returns the numpy.ndarray containing the matrix
    return (correlation_mat)

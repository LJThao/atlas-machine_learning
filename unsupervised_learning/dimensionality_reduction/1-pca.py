#!/usr/bin/env python3
"""PCA V2 Module"""
import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X

    """
    # adjust the data of the mean to 0
    X -= np.mean(X, axis=0)

    # perform SVD
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    # setting T
    T = np.dot(X, Vt.T[:, :ndim])

    # returns the transformed version
    return T

#!/usr/bin/env python3
"""PCA Module"""
import numpy as np


def pca(X, var=0.95):
    """Function that performs PCA on a data set

    X is a numpy.ndarray of shape (n, d) where:
    n is the number of data points
    d is the number of dimensions in each point
    all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation
    should maintain
    Returns: the weights matrix, W, that maintains var fraction of
    X's original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new
    dimensionality of the transformed X

    """
    # perform SVD
    _, S, Vt = np.linalg.svd(X, full_matrices=False)

    # calculate cumulative variance
    cum_var = np.cumsum(S) / np.sum(S)

    # find number of components for variance
    nd = np.searchsorted(cum_var, var) + 1

    # setting W
    W = Vt.T[:, :nd]

    # return the weights matrix
    return W

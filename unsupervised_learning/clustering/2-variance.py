#!/usr/bin/env python3
"""Variance Module"""
import numpy as np


def variance(X, C):
    """Function that calculates the total intra-cluster
    variance for a data set:

    X is a numpy.ndarray of shape (n, d) containing the data set
    C is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    Returns: total variance, or None on failure

    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray):
        return None
    if X.ndim != 2 or C.ndim != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    centroids = np.linalg.norm(X[:, np.newaxis] - C, axis=2)

    total_variance = np.sum(np.min(centroids, axis=1) ** 2)

    return total_variance

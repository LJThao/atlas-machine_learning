#!/usr/bin/env python3
"""Initialize K-means Module"""
import numpy as np


def initialize(X, k):
    """Function that initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    that will be used for K-means clustering
    n is the number of data points
    d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the
    initialized centroids for each cluster, or None on failure

    """
    # validating X, k, number of data points in X
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or not (1 <= k <= X.shape[0])):
        return None

    # randomly generating k centroids using uniform distribution
    centroids = np.random.uniform(low=X.min(axis=0),
                             high=X.max(axis=0),
                             size=(k, X.shape[1]))

    # returns initialized centroids of each cluster
    return centroids

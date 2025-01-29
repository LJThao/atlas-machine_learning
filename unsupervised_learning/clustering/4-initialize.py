#!/usr/bin/env python3
"""Initialize GMM Module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Function that initializes variables for a Gaussian Mixture
    Model:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    Returns: pi, m, S, or None, None, None on failure

    """
    # validating all inputs parameters
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(k, int) or k <= 0 or k > X.shape[0]):
        return None, None, None

    d = X.shape[1]

    # init means using k-means
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    # init covariance matrices
    S = np.eye(d)[None, :, :] * np.ones((k, 1, 1))

    # init priors equally
    pi = np.full(k, 1 / k)

    return pi, m, S

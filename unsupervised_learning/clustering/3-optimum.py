#!/usr/bin/env python3
"""Optimize k Module"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Function that tests for the optimum number of clusters by
    variance:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
    clusters to check for (inclusive)

    Returns: results, d_vars, or None, None on failure
    - results is a list containing the outputs of K-means
    for each cluster size
    - d_vars is a list containing the difference in variance
    from the smallest cluster size for each cluster size

    """
    # validating all the input parameters
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(kmin, int) or kmin <= 0 or
            (kmax is not None and (not isinstance(kmax, int) or kmax < kmin)) or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    # setting kmax value
    kmax = kmax or X.shape[0]

    results, d_vars = [], []
    base_var = None

    # iterating over the range
    for k in range(kmin, kmax + 1):
        centroids, labels = kmeans(X, k, iterations)
        if centroids is None:
            return None, None

        c_var = variance(X, centroids)
        results.append((centroids, labels))

        if base_var is None:
            base_var = c_var
            d_vars.append(0.0)
        else:
            d_vars.append(abs(base_var - c_var))

    return results, d_vars

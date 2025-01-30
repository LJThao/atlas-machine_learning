#!/usr/bin/env python3
"""Maximization Module"""
import numpy as np


def maximization(X, g):
    """Function that calculates the maximization step in the
    EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    Nk is total weight
    pi is priors for each cluster
    m is centroid means for each cluster
    S is covariance matrices for each cluster
    Returns: pi, m, S, or None, None, None on failure

    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(g, np.ndarray) or g.ndim != 2 or
            X.shape[0] != g.shape[1] or
            not np.allclose(np.sum(g, axis=0), 1)):
        return None, None, None

    n, d = X.shape
    Nk = np.sum(g, axis=1, keepdims=True)

    pi, m = Nk[:, 0] / n, (g @ X) / Nk

    S = np.array([
        ((g[i, :, np.newaxis] * (X - m[i])).T @ (X - m[i])) / Nk[i]
        for i in range(g.shape[0])
    ])

    return pi, m, S

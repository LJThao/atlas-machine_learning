#!/usr/bin/env python3
"""Expectation Module"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Function that calculates the expectation step in
    the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for
    each cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the
    covariance matrices for each cluster
    Returns: gamma, log_l, or None, None on failure

    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or
            not isinstance(S, np.ndarray) or S.ndim != 3 or
            X.shape[1] != m.shape[1] or
            m.shape[0] != pi.shape[0] or S.shape[0] != pi.shape[0] or
            S.shape[1] != S.shape[2] or S.shape[1] != X.shape[1] or
            not np.isclose(np.sum(pi), 1)):
        return None, None

    k = pi.shape[0]

    gamma = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])

    if gamma is None or np.any(gamma == None):
        return None, None

    total_prob = np.sum(gamma, axis=0, keepdims=True)

    total_prob = np.maximum(total_prob, 1e-300)

    log_l = np.sum(np.log(total_prob))

    gamma /= total_prob

    return gamma, log_l

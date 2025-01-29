#!/usr/bin/env python3
"""PDF Module"""
import numpy as np


def pdf(X, m, S):
    """Function that calculates the probability density
    function of a Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the
    data points whose PDF should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean
    of the distribution
    S is a numpy.ndarray of shape (d, d) containing the
    covariance of the distribution
    Returns: P, or None on failure

    """
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(m, np.ndarray) or m.ndim != 1 or
            not isinstance(S, np.ndarray) or S.ndim != 2 or
            X.shape[1] != m.shape[0] or m.shape[0] != S.shape[0] or
            S.shape[0] != S.shape[1]):
        return None

    d = X.shape[1]

    try:
        det_S = np.linalg.det(S)
        inv_S = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return None

    n = 1 / (np.sqrt((2 * np.pi) ** d * det_S))

    diff = X - m
    exponent = -0.5 * np.sum(diff @ inv_S * diff, axis=1)

    P = n * np.exp(exponent)

    return P

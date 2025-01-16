#!/usr/bin/env python3
"""Mean Conv (X) Module"""
import numpy as np


def mean_cov(X):
    """

    X is a numpy.ndarray of shape (n, d) containing the data set:
    n is the number of data points
    d is the number of dimensions in each data point
    If X is not a 2D numpy.ndarray, raise a TypeError with the
    message X must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the message X
    must contain multiple data points
    Returns: mean, cov:
    mean is a numpy.ndarray of shape (1, d) containing the mean
    of the data set
    cov is a numpy.ndarray of shape (d, d) containing the covariance
    matrix of the data set
    You are not allowed to use the function numpy.cov

    """
    # validate if x is a 2D
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # calculate the mean
    mean = np.sum(X, axis=0, keepdims=True) / n

    # calculate the covariance matrix
    X_centered = X - mean
    cov = (X_centered.T @ X_centered) / (n - 1)

    return (mean, cov)

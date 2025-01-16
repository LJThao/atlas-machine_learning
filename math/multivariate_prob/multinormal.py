#!/usr/bin/env python3
"""MultiNormal Module"""
import numpy as np


class MultiNormal():
    """Class that that represents a Multivariate Normal
    distribution:

    data is a numpy.ndarray of shape (d, n) containing the data
    set:
    n is the number of data points
    d is the number of dimensions in each data point
    If data is not a 2D numpy.ndarray, raise a
    TypeError with the message data must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the
    message data must contain multiple data points
    data must contain multiple data points
    Set the public instance variables:
    mean - a numpy.ndarray of shape (d, 1) containing the mean
    of data
    cov - a numpy.ndarray of shape (d, d) containing the
    covariance matrix data
    You are not allowed to use the function numpy.cov

    """
    def __init__(self, data):
        """Class Constructor - init mean and covariance matrix"""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        mean_adj = data - self.mean
        self.cov = mean_adj @ mean_adj.T / (n - 1)

    def pdf(self, x):
        """Public instance method that calculates the PDF at a data
        point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.shape != self.mean.shape:
            d_shape = self.mean.shape[0]
            raise ValueError(f"x must have the shape ({d_shape}, 1)")

        d = self.mean.shape[0]
        diff = x - self.mean
        e = np.exp(-0.5 * diff.T @ np.linalg.inv(self.cov) @ diff)
        n = ((2 * np.pi) ** (d / 2) * np.linalg.det(self.cov) ** 0.5)

        return float(e / n)

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

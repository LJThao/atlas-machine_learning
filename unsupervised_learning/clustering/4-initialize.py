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

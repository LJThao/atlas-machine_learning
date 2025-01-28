#!/usr/bin/env python3
"""EM Module"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000,
        tol=1e-5, verbose=False):
    """Function that finds the best number of clusters for
    a GMM using the Bayesian Information Criterion:

    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
    clusters to check for (inclusive)
    Returns: best_k, best_result, l, b, or None, None, None, None
    on failure

    """

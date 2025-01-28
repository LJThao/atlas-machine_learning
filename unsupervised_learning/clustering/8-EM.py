#!/usr/bin/env python3
"""EM Module"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """Functionthat performs the expectation maximization for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    Returns: pi, m, S, g, l, or None, None, None, None, None on
    failure

    """

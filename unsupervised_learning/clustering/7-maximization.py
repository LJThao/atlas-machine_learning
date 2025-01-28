#!/usr/bin/env python3
"""Maximization Module"""
import numpy as np


def maximization(X, g):
    """Function that calculates the maximization step in the
    EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    Returns: pi, m, S, or None, None, None on failure

    """

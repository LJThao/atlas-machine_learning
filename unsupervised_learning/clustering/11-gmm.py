#!/usr/bin/env python3
"""GMM Module"""
import sklearn.mixture


def gmm(X, k):
    """Function that calculates a GMM from a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: pi, m, S, clss, bic

    """

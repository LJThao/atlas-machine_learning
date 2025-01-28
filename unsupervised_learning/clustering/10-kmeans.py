#!/usr/bin/env python3
"""K-means Module"""
import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss

    """

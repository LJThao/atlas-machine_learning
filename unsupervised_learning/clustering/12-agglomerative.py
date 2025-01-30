#!/usr/bin/env python3
"""Agglomerative Module"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Function that performs agglomerative clustering on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    clss contains the cluster indices for each data points
    Returns: clss, a numpy.ndarray of shape (n,) containing the
    cluster indices for each data point

    """
    ward = scipy.cluster.hierarchy.linkage(X,'ward')

    clss = scipy.cluster.hierarchy.fcluster(ward, dist,'distance')

    scipy.cluster.hierarchy.dendrogram(ward, color_threshold=dist)

    plt.show()

    return clss

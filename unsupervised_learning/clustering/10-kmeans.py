#!/usr/bin/env python3
"""K-means Module"""
import sklearn.cluster


def kmeans(X, k):
    """Function that performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    Returns: C, clss
    - C contains centroid means for each cluster
    - clss contains the index of the cluster in C

    """
    # init and and fit the k-means model
    model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto').fit(X)

    # extracting the clusters
    C, clss = model.cluster_centers_, model.labels_

    return C, clss

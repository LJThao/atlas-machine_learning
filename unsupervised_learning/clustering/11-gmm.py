#!/usr/bin/env python3
"""GMM Module"""
import sklearn.mixture


def gmm(X, k):
    """Function that calculates a GMM from a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    pi is priors for each cluster
    m is centroid means for each cluster
    S is covariance matrices for each cluster
    clss contain the cluster indices for each data point
    bic contain the BIC value for each cluster size tested
    Returns: pi, m, S, clss, bic

    """
    model = sklearn.mixture.GaussianMixture(k)
    model.fit(X)

    pi, m, S = model.weights_, model.means_, model.covariances_
    clss, bic = model.predict(X), model.bic(X)

    return pi, m, S, clss, bic

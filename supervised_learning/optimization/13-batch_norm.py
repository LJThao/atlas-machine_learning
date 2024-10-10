#!/usr/bin/env python3
"""Batch Normalization Function"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Function that normalizes an unactivated output of a neural
    network using batch normalization:

    Z = a numpy.ndarray of shape (m, n) that should be normalized
        -> m = the number of data points
        -> n = the number of features in Z
    gamma = numpy.ndarray of shape (1, n) containing the scales used
    for batch normalization
    beta = numpy.ndarray of shape (1, n) containing the offsets used
    for batch normalization
    epsilon = a small number used to avoid division by zero

    """
    # using batch normalization, calculate the mean and var
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    # normalize the input Z
    z_norm = (Z - mean) / np.sqrt(var + epsilon)
    # then, use gamma to scale Z and beta to shift Z
    norm_mat_Z = gamma * z_norm + beta

    # returns the normalized Z matrix
    return (norm_mat_Z)

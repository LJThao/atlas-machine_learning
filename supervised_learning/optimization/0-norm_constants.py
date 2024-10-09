#!/usr/bin/env python3
"""Normalization Constants Function"""
import numpy as np


def normalization_constants(X):
    """Function that calculates the normalization (standardization)
    constants of a matrix:
    
    X = the numpy.ndarray of shape (m, nx) to normalize
    ->    m = number of data points
    ->    nx = number of features
    
    """
    # calculates the mean of each column in the matrix
    mean = np.mean(X, axis=0)
    # calculates the std of each column
    std = np.std(X, axis=0)

    # returns the mean and std of each column
    return (mean, std)

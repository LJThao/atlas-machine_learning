#!/usr/bin/env python3
"""Normalize Function"""
import numpy as np


def normalize(X, m, s):
    """Function that normalizes(standardizes) a matrix:
    
    X = the numpy.ndarray of shape (d, nx) to normalize
    ->    d = number of data points
    ->    nx = number of features
    m = numpy.ndarray of shape (nx,), contains the mean of all features of X
    s = numpy.ndarray of shape (nx,), contains the standard deviation of all
    features of X
    
    """
    # returns the normalized X matrix
    return (X - m) / s

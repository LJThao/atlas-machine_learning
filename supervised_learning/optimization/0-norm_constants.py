#!/usr/bin/env python3
"""Normalization Constants Function"""
import numpy as np


def normalization_constants(X, m, s):
    """Function that normalizes(standardizes) a matrix:
    
    X = the numpy.ndarray of shape (d, nx) to normalize
    ->    d = number of data points
    ->    nx = number of features
    m = numpy.ndarray of shape (nx,), contains the mean of all features of X
    s = numpy.ndarray of shape (nx,), contains the standard deviation of all
    features of X
    
    """
    
#!/usr/bin/env python3
"""Shuffle Function"""
import numpy as np


def shuffle_data(X, Y):
    """Function that shuffles the data points in two
    matrices the same way:

    X = the first numpy.ndarray of shape (m, nx) to shuffle
        -> m = the number of data points
        -> nx = the number of features in X
    Y = the second numpy.ndarray of shape (m, ny) to shuffle
        -> m = the same number of data points as in X
        -> ny = the number of features in Y

    """
    
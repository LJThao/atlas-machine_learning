#!/usr/bin/env python3
"""Mini-Batch Function"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Function that that creates mini-batches to be used for training
    a neural network using mini-batch gradient descent:

    X = a numpy.ndarray of shape (m, nx) representing input data
        -> m = the number of data points
        -> nx = the number of features in X
    Y = a numpy.ndarray of shape (m, ny) representing the labels
        -> m = the same number of data points as in X
        -> ny = the number of classes for classification tasks.
    batch_size = the number of data points in a batch

    """

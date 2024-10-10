#!/usr/bin/env python3
"""Mini-Batch Function"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Function that creates mini-batches to be used for training
    a neural network using mini-batch gradient descent:

    X = a numpy.ndarray of shape (m, nx) representing input data
        -> m = the number of data points
        -> nx = the number of features in X
    Y = a numpy.ndarray of shape (m, ny) representing the labels
        -> m = the same number of data points as in X
        -> ny = the number of classes for classification tasks.
    batch_size = the number of data points in a batch

    """
    # shuffling the data points
    X, Y = shuffle_data(X, Y)
    # initialize an empty list, setting m for the data points
    mini_batches = []
    m = X.shape[0]
    # for loop for creating mini-batches
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    # returns the list of mini-batches containing tuples (X_batch, Y_batch)
    return (mini_batches)

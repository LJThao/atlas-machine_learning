#!/usr/bin/env python3
"""Transfer Knowledge Module"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """Function that pre-processes the data for your model:

    X = a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
    where m is the number of data points
    Y = a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    X_p = a numpy.ndarray containing the preprocessed X
    Y_p = a numpy.ndarray containing the preprocessed Y

    """

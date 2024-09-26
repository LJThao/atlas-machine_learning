#!/usr/bin/env python3
"""One Hot Encode function"""
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric label vector
    into a one-hot matrix:
    
    Y = numpy.ndarray with shape (m,) containing numeric
    class labels
    classes = max number of classes in Y

    """
    # checks Y for a 1-dim array, if not return
    if not isinstance(Y, np.ndarray):
        return None
    # checks Y if it doesn't have 1 dim, if true return
    if Y.ndim != 1:
        return None
    # checks if classes is an int, if not return
    if not isinstance(classes, int):
        return None
    # checks if classes is less than or equal to 0, if true return
    if classes <= 0:
        return None

    # use the numpy broadcasting rule to return a one-hot encoding of Y
    oh_matrix = (np.arange(classes)[:, None] == Y).astype(int)

    return oh_matrix

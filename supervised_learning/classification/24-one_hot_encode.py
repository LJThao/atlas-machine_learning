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
    # checks Y for a 1-dim array
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    # checks if classes is an int and positive
    if not isinstance(classes, int) or classes < 2:
        return None

    # use the numpy broadcasting rule to return a one-hot matrix
    oh_matrix = (np.arange(classes)[:, None] == Y).astype(float)

    return oh_matrix

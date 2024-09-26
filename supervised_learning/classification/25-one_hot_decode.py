#!/usr/bin/env python3
"""One Hot Decode(One Hot) function"""
import numpy as np


def one_hot_decode(one_hot):
    """function that converts a one-hot matrix into
    a vector of labels
    
    one_hot = one-hot encoded numpy.ndarray with shape (classes, m)
    classes = max number of classes
    m = number of examples

    """
    # checks one_hot for a 2-dim array
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    # use np.nonzero method to find the vector of labels
    v_labels = one_hot.nonzero()[0]

    return v_labels

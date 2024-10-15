#!/usr/bin/env python3
"""Precision Function"""
import numpy as np


def precision(confusion):
    """Function that calculates the precision for each class in a
    confusion matrix:

    confusion = a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent
    the predicted labels
        -> classes is the number of classes

    """
    # getting the true positives for each class diagonally from the matrix
    true_pos = np.diag(confusion)
    # calculating the false positives: summing then subtract the true positives
    false_pos = np.sum(confusion, axis=0) - true_pos
    # calculating the precision for each class
    precision = true_pos / (true_pos + false_pos)

    # returns shape (classes,) containing the precision of each class
    return (precision)

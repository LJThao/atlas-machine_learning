#!/usr/bin/env python3
"""Sensitivity Function"""
import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity for each class
    in a confusion matrix:

    confusion = a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
        -> classes = the number of classes

    """
    # getting the true positives for each diagonally from the matrix
    true_pos = np.diag(confusion)
    # calculating the false negatives: summing then subtract the true positives
    false_neg = np.sum(confusion, axis=1) - true_pos
    # calculating the sensitivity for each class
    sensitivity = true_pos / (true_pos + false_neg)

    # returns a shape (classes,) that contains the sensitivity of each class
    return (sensitivity)

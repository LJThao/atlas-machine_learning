#!/usr/bin/env python3
"""Specificity Function"""
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for each class
    in a confusion matrix:

    confusion = a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the
    predicted labels
        -> classes = the number of classes

    """
    # getting the true positives for each class diagonally from the matrix
    true_pos = np.diag(confusion)
    # calculating the false positives: sum then subtract the true positives
    false_pos = np.sum(confusion, axis=0) - true_pos
    # calculating the false negatives for each class
    false_neg = np.sum(confusion, axis=1) - true_pos
    # calculating the true negatives for each class
    true_neg = np.sum(confusion) - (true_pos + false_neg + false_pos)
    # calculating the specificity for each class
    specificity = true_neg / (true_neg + false_pos)

    # returns the shape (classes,) containing the specificity of each class
    return (specificity)

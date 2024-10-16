#!/usr/bin/env python3
"""F1 Score Module"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Function that calculates the F1 score of a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes,
    classes) where row indices represent the correct labels and
    column indices represent the predicted labels
    classes = the number of classes

    """
    # calculating the sensitivity for each class
    sensi = sensitivity(confusion)
    # calculating the precision for each class
    preci = precision(confusion)
    # calculating the F1 score for each class
    f1_score = 2 * (preci * sensi) / (preci + sensi)

    # returns shape (classes,) containing the F1 score of each class
    return (f1_score)

#!/usr/bin/env python3
"""F1 Score Function"""
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

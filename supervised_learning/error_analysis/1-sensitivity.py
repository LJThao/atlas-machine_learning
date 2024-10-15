#!/usr/bin/env python3
"""Sensitivity Function"""
import numpy as np


def sensitivity(confusion):
    """Function that calculates the sensitivity for each class
    in a confusion matrix:

    confusion = a confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
        -> classes is the number of classes

    """
    
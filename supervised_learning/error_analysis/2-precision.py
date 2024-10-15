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
    
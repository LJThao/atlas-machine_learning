#!/usr/bin/env python3
"""Specificity Function"""
import numpy as np


def specificity(confusion):
    """Function that calculates the specificity for each class
    in a confusion matrix:

    confusion is a confusion numpy.ndarray of shape (classes, classes) where
    row indices represent the correct labels and column indices represent the predicted labels
        -> classes is the number of classes

    """
    
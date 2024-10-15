#!/usr/bin/env python3
"""Confusion Matrix Function"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """Function that creates a confusion matrix:

    labels = a one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point
        m = the number of data points
        classes = the number of classes
    logits = a one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels

    """
    
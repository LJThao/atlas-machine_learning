#!/usr/bin/env python3
"""One Hot Module"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Function that converts a label vector into a one-hot matrix:

    The last dimension of the one-hot matrix must be the number of classes
    and returns one-hot matrix

    """


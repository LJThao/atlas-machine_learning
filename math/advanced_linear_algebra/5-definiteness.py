#!/usr/bin/env python3
"""Definiteness Matrix Module"""
import numpy as np


def definiteness(matrix):
    """

    matrix is a numpy.ndarray of shape (n, n) whose definiteness
    should be calculated
    If matrix is not a numpy.ndarray, raise a TypeError with the
    message matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None
    Return: the string Positive definite, Positive semi-definite,
    Negative semi-definite, Negative definite, or Indefinite if the
    matrix is positive definite, positive semi-definite, negative
    semi-definite, negative definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None
    You may import numpy as np

    """
    # validating matrix
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # testing if the matrix is a square and symmetric
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # computing eigen values
    eigen_val = np.linalg.eigvalsh(matrix)

    # classifying definiteness
    if np.all(eigen_val > 0):
        return "Positive definite"
    elif np.all(eigen_val >= 0):
        return "Positive semi-definite"
    elif np.all(eigen_val < 0):
        return "Negative definite"
    elif np.all(eigen_val <= 0):
        return "Negative semi-definite"
    elif np.any(eigen_val > 0) and np.any(eigen_val < 0):
        return "Indefinite"

    # returns none if matrix doesn't fit any categories
    return None
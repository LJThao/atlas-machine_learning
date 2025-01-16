#!/usr/bin/env python3
"""Inverse Matrix Module"""


def inverse(matrix):
    """

    matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
    the message matrix must be a non-empty square matrix
    Returns: the inverse of matrix, or None if matrix is singular

    """
    
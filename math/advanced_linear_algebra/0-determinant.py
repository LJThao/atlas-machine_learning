#!/usr/bin/env python3
"""Determinant Module"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix

    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the
    message matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message
    matrix must be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix

    """
    
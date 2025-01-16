#!/usr/bin/env python3
"""Determinant Matrix Module"""


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
    # handles a 0x0 matrix
    if matrix == [[]]:
        return (1)

    # validating the input is a list of lists
    if (
        matrix and isinstance(matrix, list)
        and all(isinstance(row, list) for row in matrix)
    ):
        # make sure the matrix is a square
        size = len(matrix)
        for row in matrix:
            if len(row) != size:
                raise ValueError("matrix must be a square matrix")

        # 1x1 matrix
        if size == 1:
            return matrix[0][0]
        # 2x2 matrix
        elif size == 2:
            return (
                (matrix[0][0] * matrix[1][1]) -
                (matrix[0][1] * matrix[1][0])
            )
        else:
            # recursive for matrices
            det_val = 0
            for column_index in range(size):
                # generates the minor matrix
                minor = [
                    [
                        matrix[row][col]
                        for col in range(size) if col != column_index
                    ]
                    for row in range(1, size)
                ]
                # addition and subtraction for cofactors
                cofact = matrix[0][column_index] * determinant(minor)
                if column_index % 2 == 0:
                    det_val += cofact
                else:
                    det_val -= cofact
            return (det_val)

    else:
        # raise an error if the input is not valid
        raise TypeError("matrix must be a list of lists")

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
    # validating the matrix
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    size = len(matrix)
    if size == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # calculate the determinant matrix
    det_mat = determinant(matrix)
    if det_mat == 0:
        return None

    # calculating the adjugate matrix
    adj = adjugate(matrix)

    # calculating the inverse matrix
    inverse_matrix = [
        [adj[row][col] / det_mat for col in range(size)]
        for row in range(size)
    ]

    # returns the inverse matrix
    return (inverse_matrix)


def adjugate(matrix):
    """

    matrix is a list of lists whose adjugate matrix should be
    calculated
    If matrix is not a list of lists, raise a TypeError with
    the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with
    the message matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix

    """
    # validating the matrix
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    size = len(matrix)
    if size == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if any(len(row) != size for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # calculate the cofactor matrix
    cofactor_matrix = cofactor(matrix)

    # transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = [
        [cofactor_matrix[row][col] for row in range(size)]
        for col in range(size)
    ]

    # returns adjugate matrix
    return (adjugate_matrix)


def cofactor(matrix):
    """

    matrix is a list of lists whose cofactor matrix should be
    calculated
    If matrix is not a list of lists, raise a TypeError with
    the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
    with the message matrix must be a non-empty square matrix
    Returns: the cofactor matrix of matrix

    """
    # calculate minor
    minor_mat = minor(matrix)

    # determine the minor size
    size = len(minor_mat)
    cofactor_matrix = []

    # iterates through the rows
    for row_index in range(size):
        cofactor_row = []
        for col_index in range(size):
            sign = (-1) ** (row_index + col_index)
            cofactor_row.append(sign * minor_mat[row_index][col_index])
        cofactor_matrix.append(cofactor_row)

    # returns cofactor matrix
    return (cofactor_matrix)


def minor(matrix):
    """

    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with
    the message matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
    with the message matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix

    """
    # validating the matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # validating the matrix is square and non-empty
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    size = len(matrix)
    if size == 1:
        return [[1]]

    # calculate the minor
    minor_matrix = []
    for i in range(size):
        row_minors = []
        for j in range(size):
            # generates the sub matrix
            sub_matrix = [
                [matrix[row][col] for col in range(size) if col != j]
                for row in range(size) if row != i
            ]
            # append the determinant
            row_minors.append(determinant(sub_matrix))
        minor_matrix.append(row_minors)

    return (minor_matrix)


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

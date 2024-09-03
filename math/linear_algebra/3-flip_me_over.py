#!/usr/bin/env python3
"""Function for Matrix_Transpose"""


def matrix_transpose(matrix):
    """Returns the transpose of a 2D matrix"""
    new_matrix = []
    # iterates to get the columns
    for i in range(len(matrix[0])):
        row = []
        # iterates to get the rows and transposed
        for j in range(len(matrix)):
            row.append(matrix[j][i])
        new_matrix.append(row)
    return new_matrix

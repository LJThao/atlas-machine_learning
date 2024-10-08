#!/usr/bin/env python3
"""Function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if matrix_shape(mat1)[1] != matrix_shape(mat2)[0]:
        return None
    else:
        mat3 = []
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat2[0])):
                val = 0
                for k in range(len(mat2)):
                    val += mat1[i][k] * mat2[k][j]
                row.append(val)
            mat3.append(row.copy())
        return mat3


def matrix_shape(matrix):
    """Returns the list of the matrix"""
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    except Exception:
        return []

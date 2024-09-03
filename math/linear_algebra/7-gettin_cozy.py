#!/usr/bin/env python3
"""Function that concatenates two matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices on a specific axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            mat3 = []
            for row in mat1:
                mat3.append(row.copy())
            for i in range(len(mat2)):
                mat3.append(mat2[i].copy())
            return mat3
    else:
        if len(mat1) != len(mat2):
            return None
        else:
            mat3 = []
            for i in range(len(mat1)):
                mat3.append(mat1[i].copy() + mat2[i].copy())
            return mat3

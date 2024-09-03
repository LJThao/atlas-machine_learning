#!/usr/bin/env python3
"""Function for adding two matrices"""

def add_matrices2D(mat1, mat2):
    """Adding two matrices"""
    if len(mat1) != len(mat2):
        return None
    else:
        mat3 = []
        for i in range(len(mat1)):
            row = add_arrays(mat1[i], mat2[i])
            if row is None:
                return None
            else:
                mat3.append(row)
        return mat3

def add_arrays(arr1, arr2):
    """Adding two arrays"""
    if len(arr1) != len(arr2):
        return None
    else:
        arr3 = []
        for i in range(len(arr1)):
            arr3.append(arr1[i] + arr2[i])
        return arr3

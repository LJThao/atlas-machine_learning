#!/usr/bin/env python3
"""Calculate a Shape of a Matrix"""

def matrix_shape(matrix):
    """Returned as a list of integers"""
    # execute this to concatenates the list for the shape
    try:
        return [len(matrix)] + matrix_shape(matrix[0])
    # if try doesn't work execute this and returns empty list
    except Exception:
        return []

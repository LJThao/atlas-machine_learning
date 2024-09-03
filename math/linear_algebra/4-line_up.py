#!/usr/bin/env python3
"""Function to add two arrays"""

def add_arrays(arr1, arr2):
    """Adding two arrays element-wise"""
    if len(arr1) != len(arr2):
        return None
    else:
        arr3 = []
        for i in range(len(arr1)):
            arr3.append(arr1[i] + arr2[i])
        return arr3
 
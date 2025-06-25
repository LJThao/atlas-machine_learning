#!/usr/bin/env python3
"""From Numpy Module"""
import pandas as pd


def from_numpy(array):
    """Function that creates a pd.DataFrame from a np.ndarray:

    array is the np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order
    and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame

    """
    # create column names
    column_labels = [chr(65 + i) for i in range(array.shape[1])]

    # create dataframe
    result_df = pd.DataFrame(array, columns=column_labels)

    # returning the newly created df
    return result_df

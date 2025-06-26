#!/usr/bin/env python3
"""Flip it and Switch it Module"""


def flip_switch(df):
    """Function that takes a pd.DataFrame and:

    Sorts the data in reverse chronological order.
    Transposes the sorted dataframe.
    Returns: the transformed pd.DataFrame.

    """
    flip_rows = df[::-1].transpose()

    return flip_rows

#!/usr/bin/env python3
"""Sort Module"""


def high(df):
    """Function that takes a pd.DataFrame and:

    Sorts it by the High price in descending order.
    Returns: the sorted pd.DataFrame

    """
    sort_rows = df.sort_values(by="High", ascending=False)

    return sort_rows

#!/usr/bin/env python3
"""Indexing Module"""


def index(df):
    """Function that takes a pd.DataFrame and:

    Sets the Timestamp column as the index of the dataframe.
    Returns: the modified pd.DataFrame.

    """
    index_rows = df.set_index("Timestamp")

    return index_rows

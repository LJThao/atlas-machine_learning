#!/usr/bin/env python3
"""Hierarchy Module"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Function that takes two pd.DataFrame objects and:

    Rearranges the MultiIndex so that Timestamp is the first level.
    Concatenates the bitstamp and coinbase tables from timestamps
    1417411980 to 1417417980, inclusive.
    Adds keys to the data, labeling rows from df2 as bitstamp and
    rows from df1 as coinbase.
    Ensures the data is displayed in chronological order.
    Returns: the concatenated pd.DataFrame.

    """
    # set Timestamp as index for df
    df1 = index(df1)
    df2 = index(df2)

    # filter rows between them
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]

    # concat the filtered df with keys
    timestamp_rows = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    # reorder index
    timestamp_rows = timestamp_rows.swaplevel(0, 1)

    # sort by Timestamp
    timestamp_rows = timestamp_rows.sort_index()

    return timestamp_rows

#!/usr/bin/env python3
"""Concat Module"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Function that takes two pd.DataFrame objects and:

    Indexes both dataframes on their Timestamp columns.
    Includes all timestamps from df2 (bitstamp) up to and including
    timestamp 1417411920.
    Concatenates the selected rows from df2 to the top of df1 (coinbase).
    Adds keys to the concatenated data, labeling the rows from df2 as
    bitstamp and the rows from df1 as coinbase.
    Returns the concatenated pd.DataFrame

    """
    # set timestamp as index
    df1 = index(df1)
    df2 = index(df2)

    # filter df2 to only include 1417411920
    df2 = df2[df2.index <= 1417411920]

    # concat with keys to label the source of each row
    combined_rows = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])

    return combined_rows

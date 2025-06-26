#!/usr/bin/env python3
"""Prune Module"""


def prune(df):
    """Function that takes a pd.DataFrame and:

    Removes any entries where Close has NaN values.
    Returns: the modified pd.DataFrame.

    """
    pruned_rows = df[df["Close"].notna()]

    return pruned_rows

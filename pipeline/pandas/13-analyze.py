#!/usr/bin/env python3
"""Analyze Module"""


def analyze(df):
    """Function that takes a pd.DataFrame and:

    Computes descriptive statistics for all columns except the Timestamp
    column.
    Returns a new pd.DataFrame containing these statistics.

    """
    stats_rows = df.drop(columns=["Timestamp"]).describe()

    return stats_rows

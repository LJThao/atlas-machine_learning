#!/usr/bin/env python3
"""To Slice Module"""


def slice(df):
    """Function that takes a pd.DataFrame and:

    Extracts the columns High, Low, Close, and Volume_BTC.
    Selects every 60th row from these columns.
    Returns: the sliced pd.DataFrame

    """
    # selecting every 60th row
    sliced_rows = df.loc[::60, ["High", "Low", "Close", "Volume_(BTC)"]]

    return sliced_rows

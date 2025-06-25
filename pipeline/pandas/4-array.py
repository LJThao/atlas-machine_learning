#!/usr/bin/env python3
"""To Numpy Module"""
import pandas as pd


def array(df):
    """Function that takes a pd.DataFrame as input and performs the following:

    df is a pd.DataFrame containing columns named High and Close.
    The function should select the last 10 rows of the High and Close columns.
    Convert these selected values into a numpy.ndarray.
    Returns: the numpy.ndarray

    """
    # select the last 10 rows of the High and Close columns
    price_rows = df[["High", "Close"]].tail(10).to_numpy()

    return price_rows

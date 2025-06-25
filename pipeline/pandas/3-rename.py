#!/usr/bin/env python3
"""Rename Module"""
import pandas as pd


def rename(df):
    """Function that takes a pd.DataFrame as input and performs the following:

    df is a pd.DataFrame containing a column named Timestamp.
    The function should rename the Timestamp column to Datetime.
    Convert the timestamp values to datatime values
    Display only the Datetime and Close column
    Returns: the modified pd.DataFrame

    """
    # rename column
    df = df.rename(columns={"Timestamp": "Datetime"})

    # converting
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")

    # only keep datetime and close columns
    renamed = df[["Datetime", "Close"]]

    return renamed

#!/usr/bin/env python3
"""Fill Module"""


def fill(df):
    """Function that takes a pd.DataFrame and:

    Removes the Weighted_Price column.
    Fills missing values in the Close column with the previous row's value.
    Fills missing values in the High, Low, and Open columns with the corresponding
    Close value in the same row.
    Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Returns: the modified pd.DataFrame.

    """
    # remove the Weighted_Price column
    df = df.drop(columns=["Weighted_Price"])

    # fill missing Close values using forward fill
    df["Close"] = df["Close"].fillna(method="ffill")

    # use the same rows Close value
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df["Close"])

    # set missing volume values
    for col in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[col] = df[col].fillna(0)

    updated_rows = df

    return updated_rows

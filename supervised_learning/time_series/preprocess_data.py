#!/usr/bin/env python3
"""Preprocessing Data Module"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def preprocess(
    file=(
        '/root/atlas-machine_learning/supervised_learning/'
        'time_series/data/coinbase.csv'
    ),
    window_size=24):
    """Preprocesses the data for the time series forecasting"""
    # load the file
    df = pd.read_csv(file)

    # print columns for visual
    print("Columns in CSV:", df.columns.tolist())

    if 'Close' in df.columns:
        df = df[['Close']]
    elif 'close' in df.columns:
        df = df[['close']]
    else:
        raise ValueError("Close column is not found in the CSV.")

    # drop missing values
    df = df.dropna()

    # scale data
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # create windows
    X, y = [], []
    for i in range(len(scaled) - window_size):
        X.append(scaled[i:i + window_size])
        y.append(scaled[i + window_size])

    # save files
    np.save('X.npy', np.array(X))
    np.save('y.npy', np.array(y))
    np.save('scaler.npy', scaler)


if __name__ == '__main__':
    preprocess()

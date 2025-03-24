#!/usr/bin/env python3
"""RNN Module"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Function that performs forward propagation for a simple RNN:

    rnn_cell is an instance of RNNCell that will be used for the forward
    propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs

    """
    t, m, i = X.shape
    h = h_0.shape[1]

    # storing all hidden states then all outputs
    H = np.zeros((t + 1, m, h))
    Y = []

    H[0] = h_0
    # iterate through each time step and store output
    for t in range(t):
        H[t + 1], y = rnn_cell.forward(H[t], X[t])
        Y.append(y)

    Y = np.array(Y)

    return H, Y

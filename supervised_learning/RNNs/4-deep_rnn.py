#!/usr/bin/env python3
"""Deep RNN Module"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Function that performs forward propagation for a deep RNN:

    rnn_cells is a list of RNNCell instances of length l that will be
    used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape
    (l, m, h)
    h is the dimensionality of the hidden state
    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs

    """
    t, m, i = X.shape
    l, _, h = h_0.shape

    H = np.zeros((l, t + 1, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].by.shape[1]))

    H[:, 0] = h_0

    for t in range(t):
        x_t = X[t]
        for layer in range(l):
            h_prev = H[layer, t]
            h_next, y = rnn_cells[layer].forward(h_prev, x_t)
            H[layer, t + 1] = h_next
            x_t = h_next
        Y[t] = y

    return H, Y

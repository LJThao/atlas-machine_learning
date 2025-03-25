#!/usr/bin/env python3
"""Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Function that performs forward propagation for a bidirectional RNN:

    bi_cell is an instance of BidirectinalCell that will be used for the
    forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction, given as a
    numpy.ndarray of shape (m, h)
    h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction, given as a
    numpy.ndarray of shape (m, h)
    Returns: H, Y
    H is a numpy.ndarray containing all of the concatenated hidden states
    Y is a numpy.ndarray containing all of the outputs

    """
    t, m, i = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    Hf[0] = h_0
    Hb[-1] = h_t

    for t in range(t):
        Hf[t + 1] = bi_cell.forward(Hf[t], X[t])

    for t in range(t - 1, -1, -1):
        Hb[t] = bi_cell.backward(Hb[t + 1], X[t])

    # concatenate both forward and backward
    H = np.concatenate((Hf[1:], Hb[:-1]), axis=2)

    # compute output using bi cell output function
    Y = bi_cell.output(H)

    return H, Y

#!/usr/bin/env python3
"""Bidirectional Cell Backward Module --
based on 5-bi_forward.py"""
import numpy as np


class BidirectionalCell():
    """Class that represents a bidirectional cell of an RNN:"""
    def __init__(self, i, h, o):
        """"

        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs

        """
        # init weights and biases
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function that calculates the hidden state in the
        forward direction for one tim:

        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state

        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(h_x @ self.Whf + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """Function that calculates the hidden state in the
        backward direction for one time step:

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
        m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing the next hidden
        state

        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(h_x @ self.Whb + self.bhb)

        return h_prev

#!/usr/bin/env python3
"""RNN Cell Module"""
import numpy as np


class RNNCell():
    """Class that represents a cell of a simple RNN"""
    def __init__(self, i, h, o):
        """

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        """
        # init the weights and biases
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """

        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state

        """
        # concatenate, compute using tanh and softmax
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(h_x @ self.Wh + self.bh)
        y = np.exp(h_next @ self.Wy + self.by)

        return h_next, y / y.sum(axis=1, keepdims=True)

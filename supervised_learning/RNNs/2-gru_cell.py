#!/usr/bin/env python3
"""GRU Cell Module"""
import numpy as np


class GRUCell():
    """Class that represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        """
        # init weights and biases
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Function to perform forward propagation for one step:

        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state

        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(h_x @ self.Wz + self.bz)
        r = sigmoid(h_x @ self.Wr + self.br)
        h_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(h_r @ self.Wh + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        y = np.exp(h_next @ self.Wy + self.by)
        y = y / y.sum(axis=1, keepdims=True)

        return h_next, y


def sigmoid(x):
    """Function to update gates"""
    return 1 / (1 + np.exp(-x))

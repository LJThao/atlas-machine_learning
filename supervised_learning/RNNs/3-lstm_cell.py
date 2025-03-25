#!/usr/bin/env python3
"""LSTM Cell Module"""
import numpy as np


class LSTMCell():
    """Class that represents an LSTM unit:"""
    def __init__(self, i, h, o):
        """

        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs

        """
        # init weights and biases
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Function that performs forward propagation for one time step:

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous
        cell state

        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        f = sigmoid(h_x @ self.Wf + self.bf)
        u = sigmoid(h_x @ self.Wu + self.bu)
        c_tilde = np.tanh(h_x @ self.Wc + self.bc)
        c_next = f * c_prev + u * c_tilde
        o = sigmoid(h_x @ self.Wo + self.bo)
        h_next = o * np.tanh(c_next)
        y = np.exp(h_next @ self.Wy + self.by)
        y = y / y.sum(axis=1, keepdims=True)

        return h_next, c_next, y


def sigmoid(x):
    """Function to update gates"""
    return 1 / (1 + np.exp(-x))

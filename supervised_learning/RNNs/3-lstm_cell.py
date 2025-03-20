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

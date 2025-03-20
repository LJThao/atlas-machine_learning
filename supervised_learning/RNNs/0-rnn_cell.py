#!/user/bin/env python3
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

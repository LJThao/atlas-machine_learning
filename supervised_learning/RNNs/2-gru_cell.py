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

#!/usr/bin/env python3
"""Bidirectional Cell Output Module --
based on 6-bi_forward.py"""
import numpy as np


class BidirectionalCell():
    """Class that represents a bidirectional cell of an RNN:"""
    def __init__(self, i, h, o):
        """"

        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs

        """

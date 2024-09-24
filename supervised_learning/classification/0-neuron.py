#!/usr/bin/env python3
"""Class Neuron that defines a single
neuron performing binary classification"""
import numpy as np


class Neuron():
    """Class Neuron"""
    def __init__(self, nx):
        """Class Constructor"""
        # checks if nx is an int, if not raise
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # checks is nx is less than 1, if it is raise
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # setting up public instance attributes W, b, A
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0

        # setting up the private instance attributes for __W, __b, __A
        @property
        def W(self):
            """getter method to get W"""
            return self.__W

        @property
        def b(self):
            """getter method to get b"""
            return self.__b

        @property
        def A(self):
            """getter method to get A"""
            return self.__A

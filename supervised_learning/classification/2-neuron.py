#!/usr/bin/env python3
"""Class Neuron that defines a single
neuron performing binary classification based on
1-neuron.py"""
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

        # setting up the private instance attributes __W, __b, __A
        self.__W = np.random.normal(0, 1, size=(1, nx))
        self.__b = 0
        self.__A = 0

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

    def forward_prop(self, X):
        """public method to calculate the forward propagation
        of the neuron.

        Formula for z = W * X + b

        """
        W = self.W
        b = self.b

        # calculating z by performing matrix multiplication
        z = np.dot(W, X) + b

        # applies sigmoid activation function
        self.__A = 1 / (1 + np.exp(-z))

        # returns the activated output of the neuron
        return self.__A

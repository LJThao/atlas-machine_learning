#!/usr/bin/env python3
"""Class NeuralNetwork that defines a neural network with a hidden
layer performing binary classification based on 11-neural_network.py"""
import numpy as np


class NeuralNetwork():
    """Class NeuralNetwork"""
    def __init__(self, nx, nodes):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # setting private instance attributes
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """use getter method for W1"""
        return self.__W1

    @property
    def b1(self):
        """use getter method for b1"""
        return self.__b1

    @property
    def A1(self):
        """use getter method for a1"""
        return self.__A1

    @property
    def W2(self):
        """use getter method for w2"""
        return self.__W2

    @property
    def b2(self):
        """use getter method for b2"""
        return self.__b2

    @property
    def A2(self):
        """use getter method for a2"""
        return self.__A2

    def forward_prop(self, X):
        """calculating the forward propagation of the neural network:

        X = input data, nx = number of input features of a neuron,
        m = number of examples

        """
        self.__A1 = 1 / (1 + np.exp(-(np.dot(self.W1, X) + self.b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.dot(self.W2, self.A1) + self.b2)))
        return self.A1, self.A2

    def cost(self, Y, A):
        """calculates the cost of the model using logistic regression"""
        log_A = np.log(A)
        log_1_minus_A = np.log(1.0000001 - A)
        loss = Y * log_A + (1 - Y) * log_1_minus_A

        # calculate the average cost
        return -np.sum(loss) / A.shape[1]

    def evaluate(self, X, Y):
        """evaluates the neural networks's predictions and returns its
        predictions"""
        A = self.forward_prop(X)[1]
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

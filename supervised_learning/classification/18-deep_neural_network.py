#!/usr/bin/env python3
"""Class DeepNeuralNetwork that defines a deep neural
network performing binary classification based on
17-deep_neural_network.py"""
import numpy as np


class DeepNeuralNetwork():
    """Class DeepNeuralNetwork"""
    def __init__(self, nx, layers):
        """Class constructor"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # setting up the private instance attributes
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # initializing all weights and biases of the network
        prev_nodes = nx
        for i, nodes in enumerate(layers, 1):
            if not isinstance(nodes, int) or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            # setting variable key to the current layer index
            key = f'{i}'
            # using the He et al. method to initialize weights
            self.weights[f'W{key}'] = np.random.randn(nodes, prev_nodes
                                                      ) * np.sqrt(
                                                          2 / prev_nodes)
            # initialized to 0s and saved the weights dictionary
            self.weights[f'b{key}'] = np.zeros((nodes, 1))
            prev_nodes = nodes

    @property
    def L(self):
        """use getter method for L"""
        return self.__L

    @property
    def cache(self):
        """use getter method for cache"""
        return self.__cache

    @property
    def weights(self):
        """use getter method for weights"""
        return self.__weights

    def forward_prop(self, X):
        """calculates the forward propagation of the neural
        network"""
        # using the sigmoid function
        def sig(Z):
            """sigmoid function"""
            return 1 / (1 + np.exp(-Z))

        # X saved to the cache dictionary using key A0
        self.__cache['A0'] = X

        # iterating each layer by applying W, b, Z
        A = X
        for i in range(1, self.__L + 1):
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, A) + b
            A = sig(Z)
            self.__cache[f'A{i}'] = A

        # returning output and the cache that was updated
        return A, self.__cache

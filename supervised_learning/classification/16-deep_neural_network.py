#1/usr/bin/env python3
"""Class DeepNeuralNetwork that defines a deep neural
network performing binary classification"""
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
        for layer in layers:
            if not isinstance(layer, int) or layer <= 0:
                raise TypeError("layers must be a list of positive integers")

        # setting up the public instance attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # initializing all weights and biases of the network
        prev_nodes = nx
        for i, nodes in enumerate(layers, 1):
            # setting variable key for the current layer index
            key = f'{i}'
            # using the He et al. method to initialize weights
            self.weights[f'W{key}'] = np.random.randn(nodes, prev_nodes) * np.sqrt(2 / prev_nodes)
            # initialized to 0s and saved the weights dictionary
            self.weights[f'b{key}'] = np.zeros((nodes, 1))
            prev_nodes = nodes
    
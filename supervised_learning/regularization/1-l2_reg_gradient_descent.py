#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization Module"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization:

    Y = a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
    classes = the number of classes
    m = the number of data points
    weights = a dictionary of the weights and biases of the neural network
    cache = a dictionary of the outputs of each layer of the neural network
    alpha = the learning rate
    lambtha = the L2 regularization parameter
    L = the number of layers of the network

    """
    # gets the # of training examples from shape of the label mat Y
    m = Y.shape[1]

    # iterate from the last layer to the first to perform backprop
    for layer in range(L, 0, -1):
        # get the A and prev activations from cache
        A = cache[f'A{layer}']
        A_prev = cache[f'A{layer - 1}'] if layer > 1 else cache['A0']

        # calculate dZ, use tanh for hidden then softmax for outer layer
        dZ = (A - Y) if layer == L else dA * (1 - np.square(A))
        # calculate dW with l2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * weights[f'W{layer}']

        # updates weights and biases using gradient descent
        weights[f'W{layer}'] -= alpha * dW
        weights[f'b{layer}'] -= alpha * (np.sum(dZ, axis=1, keepdims=True) / m)

        # calculate dA for the next layer
        dA = np.matmul(weights[f'W{layer}'].T, dZ)

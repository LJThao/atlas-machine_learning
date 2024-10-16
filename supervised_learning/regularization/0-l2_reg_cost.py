#!/usr/bin/env python3
"""L2 Regularization Cost Module"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Function that calculates the cost of a neural network with L2
    regularization:

    cost = the cost of the network without L2 regularization
    lambtha = the regularization parameter
    weights = a dictionary of the weights and biases
    (numpy.ndarrays) of the neural network
    L = the number of layers in the neural network
    m = the number of data points used

    """
    # computing the sum of the squared weights the layers
    l2_sum = sum(np.sum(v ** 2) for k, v in weights.items() if 'W' in k)
    # calculate the l2 regularization cost
    reg_cost = (lambtha / (2 * m)) * l2_sum
    
    # return the cost of the network accounting of L2 regularization
    return (cost + reg_cost)

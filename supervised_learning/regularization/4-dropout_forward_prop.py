#!/usr/bin/env python3
"""Forward Propagation with Dropout Module"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Function that conducts forward propagation using Dropout:

    X = a numpy.ndarray of shape (nx, m) containing the input data for the network
    nx = the number of input features
    m = the number of data points
    weights = a dictionary of the weights and biases of the neural network
    L = the number of layers in the network
    keep_prob = the probability that a node will be kept
    -> All layers except the last should use the tanh activation function
    -> The last layer should use the softmax activation function

    """
    
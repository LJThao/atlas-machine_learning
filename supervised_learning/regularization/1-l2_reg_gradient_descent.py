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
    
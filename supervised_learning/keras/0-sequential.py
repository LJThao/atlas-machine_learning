#!/usr/bin/env python3
"""The Build Model with Keras Module"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Function that builds a neural network with the Keras library:

    nx = the number of input features to the network
    layers = a list containing the number of nodes in each layer of the
    network
    activations = a list containing the activation functions used for each
    layer of the network
    lambtha = the L2 regularization parameter
    keep_prob = the probability that a node will be kept for dropout
    """

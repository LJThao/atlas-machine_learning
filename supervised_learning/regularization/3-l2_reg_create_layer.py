#!/usr/bin/env python3
"""L2 Regularization Layer Module"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Function that creates a neural network layer in tensorFlow that
    includes L2 regularization:

    prev = a tensor containing the output of the previous layer
    n = the number of nodes the new layer should contain
    activation = the activation function that should be used on the layer
    lambtha = the L2 regularization parameter

    """
    
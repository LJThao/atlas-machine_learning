#!/usr/bin/env python3
"""Create a Layer with a Dropout Module"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob,training=True):
    """Function that creates a layer of a neural network using dropout:

    prev = a tensor containing the output of the previous layer
    n = the number of nodes the new layer should contain
    activation = the activation function for the new layer
    keep_prob = the probability that a node will be kept
    training = a boolean indicating whether the model is in training mode

    """
    
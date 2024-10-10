#!/usr/bin/env python3
"""Batch Normalization Upgraded Function"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Function that creates a batch normalization layer for
    a neural network in tensorflow:

    prev = the activated output of the previous layer
    n = the number of nodes in the layer to be created
    activation = the activation function that should be used on the
    output of the layer

    """

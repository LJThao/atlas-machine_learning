#!/usr/bin/env python3
"""Layers Function"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """returns the tensor output of the layer:

    prev = the previous layer of the tensor output
    n = number of nodes in the layer to create
    activation = activation function that the layer should
    use

    """
    # implement using the He et. al initialization for layer weights
    layer_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    # create a dense layer with n, activates, initializes the weights,
    # and name the layer
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=layer_weights,
        name='layer'
        )
    tensor_output = layer(prev)
    # returns an output of each neuron passed on to the next layer
    return tensor_output

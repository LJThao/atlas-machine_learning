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
    # adding a dense layer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(
        units=n,
        activation=None,
        kernel_initializer=init
    )(prev)
    # calculate mean and var using moments function
    mean, var = tf.nn.moments(dense_layer, axes=0)
    # incorporating trainable parameters gamma and betta as 1 and 0
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    # epsilon set
    epsilon = 1e-7
    # input tensor x set
    x = dense_layer
    # batch normalization layer
    bn_layer = tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         beta,
                                         gamma,
                                         epsilon)
    # applying activation function
    tensor = activation(bn_layer)

    # returns a tensor of the activated output for the layer
    return (tensor)

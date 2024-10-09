#!/usr/bin/env python3
"""Momentum Upgraded Function"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Function that sets up the gradient descent with momentum
    optimization algorithm in TensorFlow:

    alpha = the learning rate.
    beta1 = the momentum weight.

    """
    # creates an momentum optimizer with alpha and beta1
    optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)

    # returns optimizer
    return (optimizer)

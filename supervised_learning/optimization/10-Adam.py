#!/usr/bin/env python3
"""Adam Upgraded Function"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Function that sets up the Adam optimization algorithm
    in TensorFlow:

    alpha = the learning rate
    beta1 = the weight used for the first moment
    beta2 = the weight used for the second moment
    epsilon = a small number to avoid division by zero

    """
    # use keras API to create Adam Optimizer with the parameters
    adam_op = tf.keras.optimizers.Adam(learning_rate=alpha,
                                       beta_1=beta1,
                                       beta_2=beta2,
                                       epsilon=epsilon)

    # returns optimizer
    return (adam_op)

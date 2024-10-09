#!/usr/bin/env python3
"""RMSProp Upgraded Function"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Function that sets up the RMSProp optimization algorithm
    in TensorFlow:

    alpha = the learning rate
    beta2 = the RMSProp weight (rho/discounting factor), between 0 to 1
    epsilon = a small number to avoid division by zero

    """
    # setting up using Keras API to create RMSProp optimizer
    rms_optimizer = tf.keras.optimizers.RMSprop(learning_rate=alpha,
                                                rho=beta2,
                                                epsilon=epsilon)

    # returns optimizer
    return (rms_optimizer)

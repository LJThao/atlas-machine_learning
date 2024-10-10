#!/usr/bin/env python3
"""Learning Rate Decay Upgraded Function"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Function that creates a learning rate decay operation
    in tensorflow using inverse time decay:

    alpha = the original learning rate
    decay_rate = weight used to determine the rate at which alpha will decay
    decay_step = number of passes of gradient descent that should occur before
    alpha is decayed further

    """
    # use keras API module to create the learning rate decay op
    decayed_learning_rate = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )

    # returns the learning rate decay operation
    return (decayed_learning_rate)

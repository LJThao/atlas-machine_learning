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
    
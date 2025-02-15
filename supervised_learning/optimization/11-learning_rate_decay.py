#!/usr/bin/env python3
"""Learning Rate Decay Function"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Function that updates the learning rate using inverse time decay in
    numpy:

    alpha = the original learning rate
    decay_rate = weight used to determine the rate at which alpha will decay
    global_step = number of passes of gradient descent that have elapsed
    decay_step = number of passes of gradient descent that should occur before
    alpha is decayed further

    """
    # returns the updated value for alpha using the inverse time decay formula
    return (alpha) / (1 + decay_rate * (global_step // decay_step))

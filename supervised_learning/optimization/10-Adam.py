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
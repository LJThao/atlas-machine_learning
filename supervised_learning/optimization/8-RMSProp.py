#!/usr/bin/env python3
"""RMSProp Upgraded Function"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Function that sets up the RMSProp optimization algorithm
    in TensorFlow:

    alpha = the learning rate
    beta2 = the RMSProp weight (Discounting factor)
    epsilon = a small number to avoid division by zero

    """
    
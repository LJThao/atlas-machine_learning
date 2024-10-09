#!/usr/bin/env python3
"""Momentum Upgraded Function"""
import numpy as np


def create_momentum_op(alpha, beta1):
    """Function that sets up the gradient descent with momentum
    optimization algorithm in TensorFlow:

    alpha = the learning rate.
    beta1 = the momentum weight.

    """
    
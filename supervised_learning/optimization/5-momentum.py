#!/usr/bin/env python3
"""Momentum Function"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Function that updates a variable using the gradient descent
     with momentum optimization algorithm:

    alpha = the learning rate
    beta1 = the momentum weight
    var = a numpy.ndarray containing the variable to be updated
    grad = a numpy.ndarray containing the gradient of var
    v = the previous first moment of var

    """
    
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
    # updates the moment with momentum
    new_mom = (beta1 * v) + ((1 - beta1) * grad)
    # updates the variable with alpha and new moment v
    updated_var = var - (alpha * new_mom)

    # returns the updated variable and the new moment
    return (updated_var, new_mom)

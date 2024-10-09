#!/usr/bin/env python3
"""RMSProp Function"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Function that updates a variable using the RMSProp optimization
    algorithm:

    alpha = the learning rate
    beta2 = the RMSProp weight
    epsilon = a small number to avoid division by zero
    var = a numpy.ndarray containing the variable to be updated
    grad = a numpy.ndarray containing the gradient of var
    s = the previous second moment of var

    """
    # use Root Mean Square Propagation optimization algorithm to update the s
    new_s = (beta2 * s) + (1 - beta2) * (grad ** 2)
    # use the updated s to adjust alpha for each parameter
    updated_var = var - alpha * grad / ((new_s ** 0.5) + epsilon)

    # returns the updated variable and the new moment
    return (updated_var, new_s)

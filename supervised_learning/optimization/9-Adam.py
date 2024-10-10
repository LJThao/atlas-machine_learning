#!/usr/bin/env python3
"""Adam Function"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Function that updates a variable in place using the Adam optimization
    algorithm:

    alpha = the learning rate
    beta1 = the weight used for the first moment
    beta2 = the weight used for the second moment
    epsilon = a small number to avoid division by zero
    var = a numpy.ndarray containing the variable to be updated
    grad = a numpy.ndarray containing the gradient of var
    v = the previous first moment of var
    s = the previous second moment of var
    t = the time step used for bias correction

    """
    # updating first and second moments
    v = (beta1 * v) + (1 - beta1) * grad
    s = (beta2 * s) + (1 - beta2) * (grad ** 2)
    # bias corrections for them
    v_bias_correct = ((v / (1 - beta1 ** t)))
    s_bias_correct = ((s / (1 - beta2 ** t)))

    # updating the variable using the Adam optimization
    updated_var = var - alpha * v_bias_correct / (np.sqrt(s_bias_correct) + epsilon)

    # returns the updated variable, new first moment, new second moment
    return (updated_var, v, s)

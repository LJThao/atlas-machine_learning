#!/usr/bin/env python3
"""Absorbing Chains Module"""
import numpy as np


def absorbing(P):
    """Function that determines if a markov chain is absorbing:

    P is a is a square 2D numpy.ndarray of shape (n, n)
    representing the standard transition matrix
    P[i, j] is the probability of transitioning from state i
    to state j
    n is the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure

    """
    # validating to make sure P is a square
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return False

    # setting number of states in the Markov chain
    n = P.shape[0]
    # identifying the absorbing states
    abs_states = np.diag(P) == 1

    if np.all(abs_states):
        return True

    # iterate over states to see if any can transition to abs state
    for i in range(n):
        if abs_states[i]:
            continue
        if np.any(P[i, abs_states] > 0):
            return True

    return False

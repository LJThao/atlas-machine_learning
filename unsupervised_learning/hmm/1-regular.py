#!/usr/bin/env python3
"""Regular Chains Module"""
import numpy as np


def regular(P):
    """Function that determines the steady state probabilities of a
    regular markov chain:

    P is a is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    P[i, j] is the probability of transitioning from state i to
    state j
    n is the number of states in the markov chain
    Returns: a numpy.ndarray of shape (1, n) containing the steady
    state probabilities, or None on failure

    """
    if (not isinstance(P, np.ndarray) or P.ndim != 2 or
            P.shape[0] != P.shape[1]):
        return None

    if np.any(P <= 0) or np.any(P >= 1):
        return None

    i = 100
    trans_mat = np.linalg.matrix_power(P, i)
    state_prob = trans_mat[0, :]

    return state_prob[np.newaxis, :]

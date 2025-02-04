#!/usr/bin/env python3
"""Markov Chain Module"""
import numpy as np


def markov_chain(P, s, t=1):
    """Function that determines the probability of a markov chain
    being in a particular state after a specified number of
    iterations:

    P is a square 2D numpy.ndarray of shape (n, n) representing
    the transition matrix
    P[i, j] is the probability of transitioning from state i to
    state j
    n is the number of states in the markov chain
    s is a numpy.ndarray of shape (1, n) representing the
    probability of starting in each state
    t is the number of iterations that the markov chain has
    been through
    Returns: a numpy.ndarray of shape (1, n) representing the
    probability of being in a specific state after t iterations,
    or None on failure

    """
    # validating the input parameters
    if (
        not isinstance(P, np.ndarray)
        or not isinstance(s, np.ndarray)
        or not isinstance(t, int)
        or s.shape[1] != P.shape[0]
        or P.shape[0] != P.shape[1]
        or t < 0
    ):
        return None

    # init state probabilities
    state_prob = s

    # iterate through and update states
    for _ in range(t):
        state_prob = np.matmul(state_prob, P)

    # return specific state after t iterations
    return state_prob

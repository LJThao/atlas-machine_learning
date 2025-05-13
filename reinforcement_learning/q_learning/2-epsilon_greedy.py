#!/usr/bin/env python3
"""Epsilon Greedy Module"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Function that uses epsilon-greedy to determine the next action:

    Q is a numpy.ndarray containing the q-table
    state is the current state
    epsilon is the epsilon to use for the calculation
    You should sample p with numpy.random.uniformn to determine if your
    algorithm should explore or exploit
    If exploring, you should pick the next action with numpy.random.randint
    from all possible actions
    Returns: the next action index

    """
    # pick a random action with a probability epsilon, else pick best one
    if np.random.rand() < epsilon:
        return np.random.randint(Q.shape[1])

    action_index = np.argmax(Q[state])

    return action_index

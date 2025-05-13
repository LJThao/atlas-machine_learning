#!/usr/bin/env python3
"""Initialize Q-table Module"""
import numpy as np


def q_init(env):
    """Function that initializes the Q-table:

    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros

    """
    # init the q-table with zeros
    q_table = np.zeros((env.observation_space.n,
                        env.action_space.n))

    return q_table

#!/usr/bin/env python3
"""Train Module"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Function that implements a full training.

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    Return: all values of the score (sum of all rewards during one episode
    loop)

    """

#!/usr/bin/env python3
"""Play Module"""
import numpy as np


def play(env, Q, max_steps=100):
    """Function that has the trained agent play an episode:

    env is the FrozenLakeEnv instance
    Q is a numpy.ndarray containing the Q-table
    max_steps is the maximum number of steps in the episode

    """
    # store the board
    rendered_outputs = []

    # starting the game
    state, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        # save the board
        rendered_outputs.append(env.render())

        # choose the best action, do the action
        action = np.argmax(Q[state])
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # stop the game if it is over
        if done == True:
            break

    # save the final board
    rendered_outputs.append(env.render())

    # return the score and steps
    return total_reward, rendered_outputs

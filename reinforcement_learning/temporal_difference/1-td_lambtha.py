#!/usr/bin/env python3
"""TD(λ) Module"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Function that performs the TD(λ) algorithm:

    env is the environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action to take
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate

    """
    n_states = len(V)

    # looping through episodes
    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros(n_states)

        # stepping through the episode
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # calc the temporal difference error
            delta = reward + gamma * V[next_state] - V[state]
            E[state] += 1

            # update value function and decay traces
            for s in range(n_states):
                # skipping the holes
                if V[s] != -1.0:
                    V[s] += alpha * delta * E[s]
                    E[s] *= gamma * lambtha

            if terminated or truncated:
                break

            # moving to the next state
            state = next_state

    return V
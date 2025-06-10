#!/usr/bin/env python3
"""Monte Carlo Module"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """Function that performs the Monte Carlo algorithm:

    env is environment instance
    V is a numpy.ndarray of shape (s,) containing the value estimate
    policy is a function that takes in a state and returns the next action
    to take
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    Returns: V, the updated value estimate

    """
    # looping over the episodes, then store states and rewards
    for episode in range(episodes):
        states = []
        rewards = []

        # starting the episode
        state, _ = env.reset()

        # generate an episode, following the policy
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)

            if terminated or truncated:
                break

            # move to the next state
            state = next_state

        # init return
        G = 0

        # processing the episode backward to calculate returns
        for t in range(len(states) - 1, -1, -1):
            G = rewards[t] + gamma * G
            s = states[t]

            # determining earlier states
            if episode < len(states):
                seen_states = states[:episode]
            else:
                seen_states = states[:len(states)]

            # only update the state not visited
            if s not in seen_states and V[s] != -1.0:
                V[s] += alpha * (G - V[s])

    # return the updated estimated value
    return V

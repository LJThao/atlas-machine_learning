#!/usr/bin/env python3
"""SARSA(λ) Module"""
import numpy as np

def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs SARSA(λ):

    env is the environment instance
    Q is a numpy.ndarray of shape (s,a) containing the Q table
    lambtha is the eligibility trace factor
    episodes is the total number of episodes to train over
    max_steps is the maximum number of steps per episode
    alpha is the learning rate
    gamma is the discount rate
    epsilon is the initial threshold for epsilon greedy
    min_epsilon is the minimum value that epsilon should decay to
    epsilon_decay is the decay rate for updating epsilon between episodes
    Returns: Q, the updated Q table

    """
    for episode in range(episodes):
        # reset traces every episode
        eligibility = np.zeros_like(Q)

        # init the state and start the episode
        state, _ = env.reset()

        # choose the first action
        if np.random.uniform() < epsilon:
            action = np.random.randint(Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            # taking action
            next_state, reward, terminated, truncated, _ = env.step(action)

            # then choosing next action
            if np.random.uniform() < epsilon:
                next_action = np.random.randint(Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            # computing the TD error
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # update eligibility trace
            eligibility[state, action] += 1

            # update Q table
            Q += alpha * delta * eligibility

            # decay eligibility traces
            eligibility *= gamma * lambtha

            # move on
            state = next_state
            action = next_action

            if terminated or truncated:
                break

        # decay epsilon
        epsilon = min_epsilon + (1 - min_epsilon) * np.exp(-epsilon_decay * episode)

    # return updated Q
    return Q

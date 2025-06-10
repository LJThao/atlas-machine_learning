#!/usr/bin/env python3
"""Train Module"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Function that implements a full training.

    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    Return: all values of the score (sum of all rewards during one episode
    loop)

    """
    # init weights and store ep scores
    weights = np.random.rand(
        env.observation_space.shape[0],
        env.action_space.n
    )
    scores = []

    for ep in range(nb_episodes):
        state, _ = env.reset()
        steps = []

        done = False
        while not done:
            action, grad = policy_gradient(state, weights)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps.append((reward, grad))

        G = 0
        for reward, grad in reversed(steps):
            # computing returns and updating weights
            G = reward + gamma * G
            weights += alpha * G * grad

        # getting the total rewards for eps and logging eps results
        score = sum(r for r, _ in steps)
        scores.append(score)
        print(f"Episode: {ep} Score: {score}")

    # rendering the env every 1k eps
    if show_result and ep % 1000 == 0:
        env.render()

    return scores

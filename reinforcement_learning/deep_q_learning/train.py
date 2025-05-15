#!/usr/bin/env python3
"""Training Module"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from PIL import Image


# wrapper for gymnasium compatibility and auto fire
class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        # automatically press fire to start the game
        obs, _, terminated, truncated, _ = self.env.step(1)
        done = terminated or truncated
        return obs if not done else self.reset()

    def step(self, action):
        obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, {}

# preprocess atari frames
class AtariProcessor(Processor):
    def process_observation(self, observation):
        obs = np.array(observation)
        obs = np.squeeze(obs)
        if obs.ndim == 3 and obs.shape[0] == 84 and obs.shape[1] != 84:
            obs = np.transpose(obs, (1, 2, 0))
        if obs.ndim == 3 and obs.shape[-1] != 1:
            obs = obs[..., 0]
        img = Image.fromarray(obs).resize((84, 84)).convert('L')
        return np.array(img).astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

# building the CNN model
def build_model(nb_actions):
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, 8, strides=4, activation='relu'))
    model.add(Conv2D(64, 4, strides=2, activation='relu'))
    model.add(Conv2D(64, 3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


if __name__ == "__main__":
    env = gym.make("ALE/Breakout-v5", render_mode=None, frameskip=1)
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    env = GymCompatibilityWrapper(env)

    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=250000
    )

    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=10000,
        gamma=0.99,
        target_model_update=5000,
        train_interval=1,
        delta_clip=1.0
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    print("Training for 50,000 steps!!!")
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
    dqn.save_weights("policy.h5", overwrite=True)
    print("Training complete. Saved!!!")

#!/usr/bin/env python3
"""Play Module"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from PIL import Image


# wrapper for Gymnasium compatibility
class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        done = terminated or truncated
        return obs if not done else self.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# processor for DQN agent
class AtariProcessor(Processor):
    def process_observation(self, observation):
        obs = np.array(observation)
        obs = np.squeeze(obs)
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        if obs.ndim == 3 and obs.shape[-1] != 1:
            obs = obs[..., 0]
        img = Image.fromarray(obs).resize((84, 84)).convert('L')
        return np.array(img).astype('uint8')

    def process_state_batch(self, batch):
        return batch.astype('float32') / 255.0

# build CNN model
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
    # set up the environment
    env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    env = GymCompatibilityWrapper(env)

    # build agent
    nb_actions = env.action_space.n
    model = build_model(nb_actions)

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = GreedyQPolicy()
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        policy=policy,
        processor=processor,
        nb_steps_warmup=0,
        gamma=0.99,
        target_model_update=5000,
        train_interval=1,
        delta_clip=1.0
    )

    dqn.compile(optimizer=Adam(learning_rate=0.00025))
    dqn.load_weights("policy.h5")

    # run test episodes and track rewards
    print("Running 5 test episodes...")
    history = dqn.test(env, nb_episodes=5, visualize=False)
    rewards = history.history['episode_reward']

    with open("episode_rewards.txt", "w") as f:
        for i, r in enumerate(rewards, 1):
            f.write(f"Episode {i}: reward = {r}\n")

    print("Rewards saved in episode_rewards.txt!!!")

    # check how many episodes scored > 10
    high_scores = [r for r in rewards if r > 10]
    print(f"Episodes with rewards > 10: {len(high_scores)} / {len(rewards)}")

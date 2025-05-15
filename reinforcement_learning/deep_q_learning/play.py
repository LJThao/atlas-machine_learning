#!/usr/bin/env python3
"""Play Module"""
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordVideo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from PIL import Image


# wrapper to make gymnasium return single observation
class GymCompatibilityWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

# preprocessing for atari frames
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

# build CNN model used by the agent
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
    # setting up environment with wrappers
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, scale_obs=True)
    env = FrameStack(env, num_stack=4)
    env = RecordVideo(
        env,
        video_folder=".",
        name_prefix="video_demo",
        episode_trigger=lambda ep: True
    )
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
        target_model_update=10000,
        train_interval=4,
        delta_clip=1.0
    )
    dqn.compile(optimizer=Adam(learning_rate=0.00025))
    dqn.load_weights("policy.h5")

    # test the agent and save gameplay video
    dqn.test(env, nb_episodes=1, visualize=False)

    print("Saved video_demo-episode-0.mp4!!!")

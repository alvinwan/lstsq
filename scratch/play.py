import gym
import os
import os.path
from gym import wrappers
from typing import Tuple
import time
import cv2
import numpy as np
import glob
from collections import deque
import gym
from gym import spaces

SAVE_DIR = 'raw-atari-precompute'
RESULTS_DIR = 'raw-atari-play'
P = 15

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def _reset(self):
        return _process_frame84(self.env.reset())

def wrap_custom(env):
    return ProcessFrame84(env)

time_id = str(time.time())[-5:]
env = gym.make('SpaceInvadersNoFrameskip-v4')
env = wrappers.Monitor(wrap_custom(env), os.path.join(RESULTS_DIR, time_id), video_callable=lambda _: True)
n_episodes = 1000

V = np.load(os.path.join(SAVE_DIR, 'raw-atari-v%d.npy' % P))
W = np.load(os.path.join(SAVE_DIR, 'raw-atari-w%d.npy' % P))

episode_rewards = []
best_mean_reward = 0
for i in range(n_episodes):
    observation = env.reset()
    while True:
        obs = observation.reshape(tuple([1] + list(observation.shape)))
        featurized = obs.reshape((1, -1)).dot(V)
        action = np.argmax(featurized.dot(W))
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
            episode_reward = env.get_episode_rewards()[-1]
            episode_rewards.append(episode_reward)
            best_mean_reward = max(
                best_mean_reward, np.mean(env.get_episode_rewards()[-100:]))
            break
    print(episode_reward)
    if i % 100 == 0:
        print(i, 'Best mean reward:', best_mean_reward)

with open(os.path.join(RESULTS_DIR, 'raw-atari-pca-%d.txt' % P)) as f:
    f.write(','.join(map(str, episode_rewards)))

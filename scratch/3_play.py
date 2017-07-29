"""
Step 5
Play it forward.
"""

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
import sys

arguments = sys.argv

assert len(arguments) == 2, 'Need number of training episodes used'

SAVE_DIR = 'fc5-precompute'
RESULTS_DIR = 'fc5-play'
# P = 15
N = int(arguments[1])

import tensorflow as tf
session_config_kwargs = {
    'gpu_options': tf.GPUOptions(per_process_gpu_memory_fraction=0.4)}


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs

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
#    assert 'NoFrameskip' in env.spec.id
#    env = NoopResetEnv(env, noop_max=30)
    return ProcessFrame84(env)

time_id = str(time.time())[-5:]
env = gym.make('SpaceInvaders-v4')
env = wrappers.Monitor(wrap_custom(env), os.path.join(RESULTS_DIR, time_id), video_callable=lambda i: i % 50 == 0)
n_episodes = 1000

#V = np.load(os.path.join(SAVE_DIR, 'raw-atari-v%d.npy' % P))
# W = np.load(os.path.join(SAVE_DIR, 'fc5-w%d.npy' % P))
W = np.load(os.path.join(SAVE_DIR, 'fc5-w-%d.npy' % N))

from dqn import get_dqn
from dqn import x_to_fc5
from dqn import x_to_action
dqn = get_dqn(session_config_kwargs=session_config_kwargs)

print('Running!')
episode_rewards = []
best_mean_reward = 0
past_timesteps = []
average_reward = 0
for i in range(n_episodes):
    observation = env.reset()
    while True:
        obs = observation.reshape(tuple([1] + list(observation.shape)))
        past_timesteps.append(obs)
        if len(past_timesteps) > 4:
            past_timesteps.pop(0)
        elif len(past_timesteps) < 4:
            continue
        # featurized = obs.reshape((1, -1)).dot(V)
        x = np.concatenate(past_timesteps, axis=3)

        featurized = x_to_fc5(x, *dqn)
        # import pdb; pdb.set_trace()
        action = np.argmax(featurized.dot(W))
        # assert action == 0
        # import pdb; pdb.set_trace()
        #action = x_to_action(x, *dqn)

        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
            episode_reward = env.get_episode_rewards()[-1]
            episode_rewards.append(episode_reward)
            best_mean_reward = max(
                best_mean_reward, np.mean(env.get_episode_rewards()[-100:]))
            break
    print(episode_reward)
    if i % 100 == 0 and i > 0:
        average_reward=  np.mean(env.get_episode_rewards())
        print(i, 'Running average:', average_reward)
        print(i, 'Last mean reward:', np.mean(env.get_episode_rewards()[-100:]))
        print(i, 'Best mean reward:', best_mean_reward)

common_path = os.path.join(RESULTS_DIR, 'fc5-w-all.csv')
if not os.path.exists(common_path):
    with open(common_path, 'w') as f:
        f.write('n,average,best mean reward')

with open(common_path, 'a') as f:
    f.write('%d,%.3f,%.3f' % (N, average_reward, best_mean_reward))
print('Wrote to common results', common_path)

# path = os.path.join(RESULTS_DIR, 'fc5-pca-%d.txt' % P)
path = os.path.join(RESULTS_DIR, 'fc5-play-%d.txt' % N)
with open(path, 'w') as f:
    f.write(','.join(map(str, episode_rewards)))
print('Wrote all episode rewards to file', path)

import gym
import numpy as np
import cv2
from featurization.canny import featurize_density
import time
import sys
import os


N = 1000
# parse cli
arguments = sys.argv
model_id = '72069_canny'
env_id = 'Centipede-v0'
if len(arguments) > 1:
    model_id = arguments[1]
if len(arguments) > 2:
    env_id = arguments[2]

env = gym.make(env_id)


ls_model = np.load('/data/alvin/lstsq/compute-210x160-%s/w_%s.npy' % (env_id, model_id))


def round(x):
    return np.round(x * 100) / 100.

action = 0
obs_84 = []
#states = []
rewards = []
all_corrects = []
f = open('../compute-210x160-%s/eval_%s.txt' % (env_id, model_id), 'w')
print('Starting')
try:
  for i in range(N):
    last_obs = obs = env.reset()
    reward = episode_reward = 0
    corrects = []
    while True:
        featurized_obs = featurize_density(obs)
        action = featurized_obs.dot(ls_model).argmax()
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        last_obs = obs
        if done:
            break
    time_id = str(time.time())[-5:]
    rewards.append(episode_reward)
    print('[%d] Ep Rew:' % i, episode_reward, 'Avg Rew:', round(np.mean(rewards)), '100 Rew:', round(np.mean(rewards[-100:])))
    f.write('%d\n' % (episode_reward))
    continue
    try:
        data = np.vstack(states).astype(np.uint8)
        np.save('state-210x160-SpaceInvaders-v0/%s_%05d' % (time_id, episode_reward), data)
    except MemoryError:
        batch_size = 10000
        n, i = len(data), 0
        while n > 0:
            data = np.vstack(states[i*batch_size:(i+1)*batch_size]).astype(np.uint8)
            np.save('state-210x160-SpaceInvaders-v0/%s_%05d_%d' % (time_id, episode_reward, i), data)
            i += 1
            n -= batch_size
except Exception as e:
    print(e)
finally:
    f.close()

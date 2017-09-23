import gym
import numpy as np
import cv2
from featurization.a3c import a3c_model
import time
import sys

N = 1000
env_id = 'SpaceInvaders-v0'
if len(sys.argv) == 2:
    env_id = sys.argv[1]
print('environment id', env_id)

def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    resized = cv2.resize(img, (84, 84), interpolation=cv2.INTER_LINEAR)
    x_t = resized.astype(np.uint8)
    return x_t

model = a3c_model(load='/data/alvin/models/%s.npy' % env_id)
env = gym.make(env_id)

action = 0
obs_84 = []
states = []
for i in range(N):
    last_obs = obs = env.reset()
    reward = episode_reward = 0
    while True:
        obs_84.append(_process_frame84(obs))
        if len(obs_84) < 4:
            continue
        featurized_obs = np.concatenate(obs_84[-4:], axis=2)
        action = model([[featurized_obs]])[0].argmax()
        obs, reward, done, info = env.step(action)
        state = np.hstack((np.ravel(last_obs), action, reward))
        episode_reward += reward
        states.append(state)
        last_obs = obs
        if done:
            break
    time_id = str(time.time())[-5:]
    try:
        data = np.vstack(states).astype(np.uint8)
        fname = 'state-210x160-%s/%s_%d_%d' % (env_id, time_id, episode_reward, len(states))
        np.save(fname, data)
        print(fname)
    except MemoryError:
        batch_size = 10000
        n, i = len(data), 0
        while n > 0:
            data = np.vstack(states[i*batch_size:(i+1)*batch_size]).astype(np.uint8)
            'state-210x160-%s/%s_%d_%d_%d' % (env_id, time_id, episode_reward, len(states), i)
            np.save(fname, data)
            print(fname)
            i += 1
            n -= batch_size
    print('Reward:', episode_reward)

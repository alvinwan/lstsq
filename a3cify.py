import numpy as np
import glob
from featurization.a3c import a3c_model
import sys
import gym
import os
import cv2

# parse cli
arguments = sys.argv

N = 100
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    N = int(arguments[1])
if len(arguments) > 2:
    env_id = arguments[2]

# collect all data
env = gym.make(env_id)
data = []
for i, path in enumerate(list(sorted(glob.iglob('state-210x160-%s/*.npy' % env_id)))[:N]):
    datum = np.load(path)
    data.append(datum)
    if i % 10 == 0:
        print('loaded', i)

raw = np.concatenate(data, axis=0)

# resize all images to 84x84
samples = []
for frame in raw[:, :-2].reshape((-1,) + env.observation_space.shape):
    samples.append(cv2.resize(frame, (84, 84))[None])
samples = np.concatenate(samples, axis=0)

# check where model is
tfmodel_path = '/data/alvin/models/%s.tfmodel' % env_id
npy_path = '/data/alvin/models/%s.npy' % env_id
load_path = tfmodel_path if os.path.exists(tfmodel_path) else npy_path

# pass all frames through a3c
model = a3c_model(layer='prelu/output', load=load_path, num_actions=env.action_space.n)
fc0s = []
print('Total frames', len(samples))
for i in range(len(samples) - 3):
    frames = samples[i:i+4].reshape((4, 84, 84, 3))
    frames = np.concatenate(frames, axis=2)
    fc0 = model([[frames]])
    fc0s.append(fc0)
    if i % 1000 == 0:
        print(i)
X = np.stack(fc0s)
Y = samples[3:, -2]
assert Y.shape[0] == X.shape[0]

np.save('compute-210x160-%s/X_%d_prelu.npy' % (env_id, len(data)), X)
np.save('compute-210x160-%s/Y_%d_prelu.npy' % (env_id, len(data)), Y)

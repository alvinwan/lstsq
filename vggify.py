import numpy as np
import glob
from featurization.vgg16 import vgg16_model
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
    samples.append(cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)[None])
samples = np.concatenate(samples, axis=0)

# pass all frames through vgg16
model = vgg16_model(layer='fc8', load_path='/data/alvin/models/vgg16.npy')
fc0s = []
print('Total frames', len(samples))
for i in range(len(samples) - 3):
    frame = samples[i].reshape((1, 224, 224, 3))
    fc0 = model(frame)
    print(fc0.shape)
    fc0s.append(fc0)
    if i % 1000 == 0:
        print(i)
X = np.stack(fc0s)
Y = raw[:, -2]
assert Y.shape[0] == X.shape[0]

np.save('compute-210x160-%s/X_%d_vgg_fc8.npy' % (env_id, len(data)), X)
np.save('compute-210x160-%s/Y_%d_vgg_fc8.npy' % (env_id, len(data)), Y)

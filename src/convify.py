import numpy as np
import glob
from featurization.conv import conv
import sys
import cv2
import gym

arguments = sys.argv

N = 100
env_id = 'SpaceInvaders-v0'
patch_size = 10
pool_size = 138
if len(arguments) > 1:
    N = int(arguments[1])
if len(arguments) > 2:
    env_id = arguments[2]
if len(arguments) > 3:
    patch_size = int(arguments[3])
    pool_size = int(arguments[4])
print('Using %d for %s (patch_size: %d, pool_size: %d)' % (N, env_id, patch_size, pool_size))

env = gym.make(env_id)

data  = []
for i, path in enumerate(list(sorted(glob.iglob('state-210x160-%s/*.npy' % env_id)))[:N]):
    datum = np.load(path)
    data.append(datum)
    if i % 10 == 0:
        print('loaded', i)

samples = np.concatenate(data, axis=0)
Y = samples[:, -2]

Xs_new = []
X_raw = samples[:, :-2].reshape((-1,) + env.observation_space.shape)
for x in X_raw:
    Xs_new.append(cv2.resize(x, (160, 160), interpolation=cv2.INTER_LINEAR)[None])
X_new = np.concatenate(Xs_new, axis=0)
X_new = np.transpose(X_new, axes=(0, 3, 1, 2))
X = conv(X_new, patch_size=patch_size, pool_size=pool_size)
assert Y.shape[0] == X.shape[0]

np.save('compute-210x160-%s/X_%d_conv_%d_%d.npy' % (env_id, len(data), patch_size, pool_size), X)
np.save('compute-210x160-%s/Y_%d_conv_%d_%d.npy' % (env_id, len(data), patch_size, pool_size), Y)
print('Finished: %d_conv_%d_%d' % (env_id, len(data), patch_size, pool_size))
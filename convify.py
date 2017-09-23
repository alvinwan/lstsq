import numpy as np
import glob
from featurization.conv import conv
import sys
import scipy.misc

arguments = sys.argv

N = 100
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    N = int(arguments[1])
if len(arguments) > 2:
    env_id = arguments[2]

data  = []
for i, path in enumerate(list(sorted(glob.iglob('state-210x160-%s/*.npy' % env_id)))[:N]):
    datum = np.load(path)
    data.append(datum)
    if i % 10 == 0:
        print('loaded', i)

samples = np.concatenate(data, axis=0)
Y = samples[:, -2]

Xs_new = []
X_raw = samples[:, :-2].reshape((-1, 210, 160, 3))
for x in X_raw:
    Xs_new.append(scipy.misc.imresize(x, (160, 160), mode='nearest')[None])
X_new = np.concatenate(Xs_new, axis=0)
X_new = np.transpose(X_new, axes=(0, 3, 1, 2))
X = conv(X_new)
assert Y.shape[0] == X.shape[0]

np.save('compute-210x160-%s/X_%d_conv.npy' % (env_id, len(data)), X)
np.save('compute-210x160-%s/Y_%d_conv.npy' % (env_id, len(data)), Y)

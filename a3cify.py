import numpy as np
import glob
from a3c import a3c_model
import sys

arguments = sys.argv

N = 100
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    N = int(arguments[1])
if len(arguments) > 2:
    env_id = arguments[2]

data = []
for i, path in enumerate(list(sorted(glob.iglob('state-210x160-%s/*.npy' % env_id)))[:N]):
    datum = np.load(path)
    data.append(datum)
    if i % 10 == 0:
        print('loaded', i)

samples = np.concatenate(data, axis=0)

model = a3c_model(layer='prelu/output', load='/data/alvin/models/%s.tfmodel' % env_id)
fc0s = []
print('Total frames', len(samples))
for i in range(len(samples) - 3):

    frames = samples[i:i+4,:-2].reshape((4, 84, 84, 3))
    frames = np.concatenate(frames, axis=2)
    fc0 = model([[frames]])
    fc0s.append(fc0)
    if i % 1000 == 0:
        print(i)
X = np.stack(fc0s)
Y = samples[3:,-2]
assert Y.shape[0] == X.shape[0]

np.save('compute-210x160-%s/X_%d_prelu.npy' % (env_id, len(data)), X)
np.save('compute-210x160-%s/Y_%d_prelu.npy' % (env_id, len(data)), Y)

from featurization.canny import featurize
import numpy as np


src = '/data/alvin/lstsq/state-210x160-SpaceInvaders-v0/00235_01170_0.npy'
dest = '/data/alvin/lstsq/compute-210x160-SpaceInvaders-v0/X_%s_canny.npy'

A = np.load(src).reshape((-1, 210, 160, 3))
B = np.array([featurize(a) for a in A])

np.save(dest % len(B), B)

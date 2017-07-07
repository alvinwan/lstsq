"""Solve ls"""


import os
import os.path
import numpy as np

P = 15
SAVE_DIR = 'raw-atari-precompute'

xtx = np.load(os.path.join(SAVE_DIR, 'raw-atari-xvtxv%d-final.npy' % P))
xty = np.laod(os.path.join(SAVE_DIR, 'raw-atari-xvty%d-final.npy' % P))

w = np.linalg.inv(xtx).dot(xty)
np.save('raw-atari-w%d.npy' % P, w)

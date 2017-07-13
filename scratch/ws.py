"""Solve ls"""


import os
import os.path
import numpy as np

P = 15
SAVE_DIR = 'raw-atari-precompute'

xtx = np.load(os.path.join(SAVE_DIR, 'raw-atari-xvtxv%d-final.npy' % P))
xty = np.load(os.path.join(SAVE_DIR, 'raw-atari-xvty%d-final.npy' % P))
print(os.path.join(SAVE_DIR, 'raw-atari-xvty%d-final.npy' % P))

w = np.linalg.inv(xtx).dot(xty)
np.save(os.path.join(SAVE_DIR,'raw-atari-w%d.npy' % P), w)
import pdb
pdb.set_trace()

import os
import os.path
import numpy as np

DIR = 'raw-atari-unpacked'
SAVE_DIR = 'raw-atari-precompute'

XTX = np.load(os.path.join(SAVE_DIR, 'raw-atari-xtx-final.npy'))
mu = np.load(os.path.join(SAVE_DIR, 'raw-atari-mu-final.npy'))
XTmu = np.load(os.path.join(SAVE_DIR, 'raw-atari-xtmu-final.npy'))

X_TX_ = XTX - 2*XTmu + mu.T.dot(mu)

lambdas, vs = np.linalg.eig(X_TX_)
for i in range(15, 20):
   range = vs[:,:i]
   np.save(os.path.join(SAVE_DIR, 'raw-atari-v%d.npy' % i), range)

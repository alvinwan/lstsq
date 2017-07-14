"""mis-saved xtx"""

import numpy as np
import os
import os.path

SAVE_DIR='raw-atari-precompute'
NUM_THREADS=56
NUM_ACTIONS = 6
D=7056
P=15

xtx=np.zeros((P,NUM_ACTIONS))
for i in range(NUM_THREADS):
    xtx += np.load(os.path.join(SAVE_DIR, 'raw-atari-xty%d-%d.npy' % (P, i)))
np.save(os.path.join(SAVE_DIR, 'raw-atari-xvty%d-final' % P), xtx)

"""mis-saved xtx"""

import numpy as np
import os
import os.path

SAVE_DIR='raw-atari-precompute'
NUM_THREADS=56
D=7056

xtx=np.zeros((D,D))
for i in range(NUM_THREADS):
    xtx += np.load(os.path.join(SAVE_DIR, 'raw-atari-xtx-%d.npy' % i))
np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx-final'), xtx)

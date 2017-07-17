"""
Step 4
LS solution
w = (X^TX)^{-1}X^Ty
"""

import os
import os.path
import numpy as np
import sys

arguments = sys.argv

assert len(arguments) == 2, 'Requires number of training episodes used'

N = int(arguments[1])

XTX_FINAL = 'fc5-xtx-%d-final.npy'
XTY_FINAL = 'fc5-xty-%d-final.npy'
MODEL = 'fc5-w-%d.npy'
SAVE_DIR = 'fc5-precompute'

xtx = np.load(os.path.join(SAVE_DIR, XTX_FINAL % N))
xty = np.load(os.path.join(SAVE_DIR, XTY_FINAL % N))

w = np.linalg.pinv(xtx).dot(xty)
np.save(os.path.join(SAVE_DIR, MODEL % N), w)

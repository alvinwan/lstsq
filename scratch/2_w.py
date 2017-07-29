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

assert len(arguments) >= 2, 'Need number of training episodes'
assert len(arguments) < 3 or 'dagger' in arguments or 'dagger2' in arguments, 'Gotta add the "dagger" or "dagger2" keyword bruh! i.e., bash script.sh 1000 9000 dagger, currently: %s' % ' '.join(arguments)

DAGGER = 'dagger' in arguments
DAGGER2 = 'dagger2' in arguments
N=int(arguments[1])
N_d = int(arguments[2]) if len(arguments) >= 3 and arguments[2].isnumeric() else N
SAVE_DIR = 'spaceinvaders-precompute'
LAYER = 'prelu'

XTX_FINAL = LAYER + '-xtx-%d-final'
XTY_FINAL = LAYER + '-xty-%d-final'
W = LAYER + '-w-%d'

if DAGGER2:
    DAGGER_DIR = 'spaceinvaders-dagger2'
    SAVE_DIR = 'spaceinvaders-dagger2-precompute'
    DAGGER2_IDX = int(arguments[4])

if DAGGER:
    print('DAGGAH!!!')
    suffix = '-dagger-%d' % N_d
    if DAGGER2:
        suffix += '-' + str(DAGGER2_IDX)
    XTX_FINAL += suffix
    XTY_FINAL += suffix
    W += suffix

XTX_FINAL += '.npy'
XTY_FINAL += '.npy'
W += '.npy'

xtx = np.load(os.path.join(SAVE_DIR, XTX_FINAL % N))
xty = np.load(os.path.join(SAVE_DIR, XTY_FINAL % N))

w = np.linalg.pinv(xtx).dot(xty)
np.save(os.path.join(SAVE_DIR, W % N), w)
#import pdb
#pdb.set_trace()

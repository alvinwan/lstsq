"""converts files to int"""

import numpy as np
import glob
import time
import sys

start = 0
if len(sys.argv) == 2:
    start = int(sys.argv[1])

paths = list(sorted(glob.iglob('state-210x160-SpaceInvaders-v0/*.npy')))
for i, path in enumerate(paths[start:], start=start):
    start = time.time()
    try:
        data = np.load(path)
    except Exception as e:
        print(e)
        print('[%d] could not process "%s"' % (i, path))
        continue
    if data.dtype == np.uint8:
        print('[%d] skipping...' % i)
        continue
    data2 = data.astype(np.uint8)
    np.save(path, data2)
    print('[%d] (%d s) %s' % (i, time.time() - start, path))

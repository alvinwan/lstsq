"""
51000 1 1196.9635016918182
Segmentation fault
"""

import os
import os.path
import numpy as np
import time

start = time.time()
DIR = 'raw-atari-unpacked'
data = []
n_corrupted = 0
for i, path in enumerate(os.listdir(DIR)):
   if i % 1000 == 0:
      print(i, n_corrupted, time.time() - start)
   full_path = os.path.join(DIR, path)
   try:
          data.append(np.load(full_path))
 #         np.load(full_path)
   except Exception as e:
          n_corrupted += 1
          print(e)
print(n_corrupted)
print(time.time() - start)
t2 = time.time()
np.save('raw-atari-unpacked', np.vstack(data))
print('saving took', time.time() - t2)

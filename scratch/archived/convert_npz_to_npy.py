import glob
import numpy as np
import os
import os.path

from zipfile import BadZipFile

DIR = './raw-atari-unpacked'

os.makedirs(DIR, exist_ok=True)

from subprocess import call

while True:
  try:
    i = 0
    rows = 0
    n =  0
    for i, path in enumerate(glob.iglob('raw-atari/*.npz')):
        if i % 100 == 0 and i > 0:
           print(' * [Info] finished', i)
           break
        if os.path.exists('raw-atari-unpacked/%s' % os.path.basename(path)):
           continue
        with np.load(path) as f:
           A = f['arr_0']
           n += 1
           rows += A.shape[0]
           np.save(os.path.join(DIR, os.path.basename(path)), f['arr_0'])
    import pdb
    pdb.set_trace()
    print(i)
    break
  except BadZipFile:
    call(['rm', path])
    print(' * [Info] Bad zip file:', path)
  except AttributeError:
    call(['rm', path])
    print(' * [Info] Bad zip file:', path)

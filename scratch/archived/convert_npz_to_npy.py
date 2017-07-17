import glob
import numpy as np
import os
import os.path
import threading
import numpy as np
from zipfile import BadZipFile

DIR = './raw'
SOURCE_DIR = './rawz'
N_THREADS = 48

os.makedirs(DIR, exist_ok=True)

from subprocess import call

def run(thread_id, paths, start):
  try:
    rows = 0
    n =  0
    for i, path in enumerate(glob.iglob(os.path.join(SOURCE_DIR, '*.npz')), start=start):
        if i % 100 == 0 and i > 0:
           print(' * [Info] finished', i)
        if os.path.exists(os.path.join(DIR, os.path.basename(path))):
           continue
        with np.load(path) as f:
           A = f['arr_0']
           n += 1
           rows += A.shape[0]
           np.save(os.path.join(DIR, os.path.basename(path)), f['arr_0'])
    print(' * [Info] Finished thread %d' % thread_id)
  except BadZipFile:
    call(['rm', path])
    print(' * [Info] Bad zip file:', path)
  except AttributeError:
    call(['rm', path])
    print(' * [Info] Bad zip file:', path)


def main():
   paths = list(os.listdir(SOURCE_DIR))
   n_paths_per_thread = int(np.ceil(len(paths) / N_THREADS))
   print('Number threads:', N_THREADS)
   print('Paths per thread:', n_paths_per_thread)
   threads = []
   for i in range(N_THREADS):
      thread_paths = paths[i*n_paths_per_thread: (i+1)*n_paths_per_thread]
      thread = threading.Thread(target=run, args=(i, thread_paths, i*n_paths_per_thread))
      thread.start()
      threads.append(thread)
      if i % 14 == 0:
         print(i, 'threads started')
   print(N_THREADS, 'threads started')
   for i in range(N_THREADS):
      threads[i].join()

if __name__ == '__main__':
   main()

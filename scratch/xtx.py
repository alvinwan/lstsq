import os
import os.path
import numpy as np
import time
import threading


N_THREADS = 72
D=7056
start = time.time()
DIR = 'raw-atari-unpacked'
SAVE_DIR = 'raw-atari-precompute'
global_xtx = np.zeros((D, D))
global_xty = np.zeros((D, 1))

def run(thread_id, paths, start):
   xtx = np.zeros((D,D))
   xty = np.zeros((D, 1))
   n_corrupted = 0
   for i, path in enumerate(paths, start=start):
      if i % 100 == 0:
         print(i, n_corrupted, time.time() - start)
      if i%4000 == 0:
         t2 = time.time()
         np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx-%d-%d' % (thread_id, i)), xtx)
         np.save(os.path.join(SAVE_DIR, 'raw-atari-xty-%d-%d' % (thread_id, i)), xty)
         print('saved', i, 'in', time.time()-t2)
      full_path = os.path.join(DIR, path)
      try:
          di = np.load(full_path)
          x, y = di[:,:D], di[:,-1].reshape((-1, 1))
          xty += x.T.dot(y)
          xtx += x.T.dot(x)
 #         np.load(full_path)
      except Exception as e:
          n_corrupted += 1
          print(e)
   print(n_corrupted)
   print(time.time() - start)
   t2 = time.time()
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx-%d' % thread_id), xtx)
   print('saving took', time.time() - t2)
   global global_xtx, global_xty
   global_xtx += xtx
   global_xty += xty


def main():
   paths = list(os.listdir(DIR))
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
   global global_xtx, global_xty
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx-final'), global_xtx)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xty-final'), global_xty)

if __name__ == '__main__':
   main()
